import torch
from torch import nn
from utils import outputActivation

# The implementation of PiP architecture
class pipNet(nn.Module):

    def __init__(self, args):
        super(pipNet, self).__init__()
        self.args = args
        self.use_cuda = args.use_cuda

        # Flag for output:
        # -- Train-mode : Concatenate with true maneuver label.
        # -- Test-mode  : Concatenate with the predicted maneuver with the maximal probability.
        self.train_output_flag = args.train_output_flag
        self.use_planning = args.use_planning
        self.use_fusion = args.use_fusion

        # IO Setting
        self.grid_size = args.grid_size
        self.in_length = args.in_length
        self.out_length = args.out_length
        self.num_lat_classes = args.num_lat_classes
        self.num_lon_classes = args.num_lon_classes

        ## Sizes of network layers
        self.temporal_embedding_size = args.temporal_embedding_size
        self.encoder_size = args.encoder_size
        self.decoder_size = args.decoder_size
        self.soc_conv_depth = args.soc_conv_depth
        self.soc_conv2_depth = args.soc_conv2_depth
        self.dynamics_encoding_size = args.dynamics_encoding_size
        self.social_context_size = args.social_context_size
        self.targ_enc_size = self.social_context_size + self.dynamics_encoding_size
        self.fuse_enc_size = args.fuse_enc_size
        self.fuse_conv1_size = 2 * self.fuse_enc_size
        self.fuse_conv2_size = 4 * self.fuse_enc_size

        # Activations:
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        ## Define network parameters
        ''' Convert traj to temporal embedding'''
        self.temporalConv = nn.Conv1d(in_channels=2, out_channels=self.temporal_embedding_size, kernel_size=3, padding=1)

        ''' Encode the input temporal embedding '''
        self.nbh_lstm = nn.LSTM(input_size=self.temporal_embedding_size, hidden_size=self.encoder_size, num_layers=1)
        if self.use_planning:
            self.plan_lstm = nn.LSTM(input_size=self.temporal_embedding_size, hidden_size=self.encoder_size, num_layers=1)

        ''' Encoded dynamic to dynamics_encoding_size'''
        self.dyn_emb = nn.Linear(self.encoder_size, self.dynamics_encoding_size)

        ''' Convolutional Social Pooling on the planned vehicle and all nbrs vehicles  '''
        self.nbrs_conv_social = nn.Sequential(
            nn.Conv2d(self.encoder_size, self.soc_conv_depth, 3),
            self.leaky_relu,
            nn.MaxPool2d((3, 3), stride=2),
            nn.Conv2d(self.soc_conv_depth, self.soc_conv2_depth, (3, 1)),
            self.leaky_relu
        )
        if self.use_planning:
            self.plan_conv_social = nn.Sequential(
                nn.Conv2d(self.encoder_size, self.soc_conv_depth, 3),
                self.leaky_relu,
                nn.MaxPool2d((3, 3), stride=2),
                nn.Conv2d(self.soc_conv_depth, self.soc_conv2_depth, (3, 1)),
                self.leaky_relu
            )
            self.pool_after_merge = nn.MaxPool2d((2, 2), padding=(1, 0))
        else:
            self.pool_after_merge = nn.MaxPool2d((2, 1), padding=(1, 0))

        ''' Target Fusion Module'''
        if self.use_fusion:
            ''' Fused Structure'''
            self.fcn_conv1 = nn.Conv2d(self.targ_enc_size, self.fuse_conv1_size, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(self.fuse_conv1_size)
            self.fcn_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            self.fcn_conv2 = nn.Conv2d(self.fuse_conv1_size, self.fuse_conv2_size, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(self.fuse_conv2_size)
            self.fcn_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            self.fcn_convTrans1 = nn.ConvTranspose2d(self.fuse_conv2_size, self.fuse_conv1_size, kernel_size=3, stride=2, padding=1)
            self.back_bn1 = nn.BatchNorm2d(self.fuse_conv1_size)
            self.fcn_convTrans2 = nn.ConvTranspose2d(self.fuse_conv1_size, self.fuse_enc_size, kernel_size=3, stride=2, padding=1)
            self.back_bn2 = nn.BatchNorm2d(self.fuse_enc_size)
        else:
            self.fuse_enc_size = 0

        ''' Decoder LSTM'''
        self.op_lat = nn.Linear(self.targ_enc_size + self.fuse_enc_size,
                                self.num_lat_classes)  # output lateral maneuver.
        self.op_lon = nn.Linear(self.targ_enc_size + self.fuse_enc_size,
                                self.num_lon_classes)  # output longitudinal maneuver.
        self.dec_lstm = nn.LSTM(input_size=self.targ_enc_size + self.fuse_enc_size + self.num_lat_classes + self.num_lon_classes,
                                      hidden_size=self.decoder_size)

        ''' Output layers '''
        self.op = nn.Linear(self.decoder_size, 5)


    def forward(self, nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, lat_enc, lon_enc):

        ''' Forward target vehicle's dynamic'''
        dyn_enc = self.leaky_relu(self.temporalConv(targsHist.permute(1,2,0)))
        _, (dyn_enc, _) = self.nbh_lstm(dyn_enc.permute(2,0,1))
        dyn_enc = self.leaky_relu( self.dyn_emb(dyn_enc.view(dyn_enc.shape[1],dyn_enc.shape[2])) )

        ''' Forward neighbour vehicles'''
        nbrs_enc = self.leaky_relu(self.temporalConv(nbsHist.permute(1, 2, 0)))
        _, (nbrs_enc, _) = self.nbh_lstm(nbrs_enc.permute(2, 0, 1))
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])

        ''' Masked neighbour vehicles'''
        nbrs_grid = torch.zeros_like(nbsMask).float()
        nbrs_grid = nbrs_grid.masked_scatter_(nbsMask, nbrs_enc)
        nbrs_grid = nbrs_grid.permute(0,3,2,1)
        nbrs_grid = self.nbrs_conv_social(nbrs_grid)

        if self.use_planning:
            ''' Forward planned vehicle'''
            plan_enc = self.leaky_relu(self.temporalConv(planFut.permute(1, 2, 0)))
            _, (plan_enc, _) = self.plan_lstm(plan_enc.permute(2, 0, 1))
            plan_enc = plan_enc.view(plan_enc.shape[1], plan_enc.shape[2])

            ''' Masked planned vehicle'''
            plan_grid = torch.zeros_like(planMask).float()
            plan_grid = plan_grid.masked_scatter_(planMask, plan_enc)
            plan_grid = plan_grid.permute(0, 3, 2, 1)
            plan_grid = self.plan_conv_social(plan_grid)

            ''' Merge neighbour and planned vehicle'''
            merge_grid = torch.cat((nbrs_grid, plan_grid), dim=3)
            merge_grid = self.pool_after_merge(merge_grid)
        else:
            merge_grid = self.pool_after_merge(nbrs_grid)
        social_context = merge_grid.view(-1, self.social_context_size)

        '''Concatenate social_context (neighbors + ego's planing) and dyn_enc, then place into the targsEncMask '''
        target_enc = torch.cat((social_context, dyn_enc),1)
        target_grid = torch.zeros_like(targsEncMask).float()
        target_grid = target_grid.masked_scatter_(targsEncMask, target_enc)

        if self.use_fusion:
            '''Fully Convolutional network to get a grid to be fused'''
            fuse_conv1 = self.relu(self.fcn_conv1(target_grid.permute(0, 3, 2, 1)))
            fuse_conv1 = self.bn1(fuse_conv1)
            fuse_conv1 = self.fcn_pool1(fuse_conv1)
            fuse_conv2 = self.relu(self.fcn_conv2(fuse_conv1))
            fuse_conv2 = self.bn2(fuse_conv2)
            fuse_conv2 = self.fcn_pool2(fuse_conv2)
            # Encoder / Decoder #
            fuse_trans1 = self.relu(self.fcn_convTrans1(fuse_conv2))
            fuse_trans1 = self.back_bn1(fuse_trans1+fuse_conv1)
            fuse_trans2 = self.relu(self.fcn_convTrans2(fuse_trans1))
            fuse_trans2 = self.back_bn2(fuse_trans2)
            # Extract the location with targets
            fuse_grid_mask = targsEncMask[:,:,:,0:self.fuse_enc_size]
            fuse_grid = torch.zeros_like(fuse_grid_mask).float()
            fuse_grid = fuse_grid.masked_scatter_(fuse_grid_mask, fuse_trans2.permute(0, 3, 2, 1))

            '''Finally, Integrate everything together'''
            enc_rows_mark = targsEncMask[:,:,:,0].view(-1)
            enc_rows = [i for i in range(len(enc_rows_mark)) if enc_rows_mark[i]]
            enc = torch.cat([target_grid, fuse_grid], dim=3)
            enc = enc.view(-1, self.fuse_enc_size+self.targ_enc_size)
            enc = enc[enc_rows, :]
        else:
            enc = target_enc

        '''Maneuver recognition'''
        lat_pred = self.softmax(self.op_lat(enc))
        lon_pred = self.softmax(self.op_lon(enc))
        if self.train_output_flag:
            enc = torch.cat((enc, lat_enc, lon_enc), 1)
            fut_pred = self.decode(enc)
            return fut_pred, lat_pred, lon_pred
        else:
            fut_pred = []
            for k in range(self.num_lon_classes):
                for l in range(self.num_lat_classes):
                    lat_enc_tmp = torch.zeros_like(lat_enc)
                    lon_enc_tmp = torch.zeros_like(lon_enc)
                    lat_enc_tmp[:, l] = 1
                    lon_enc_tmp[:, k] = 1
                    # Concatenate maneuver label before feeding to decoder
                    enc_tmp = torch.cat((enc, lat_enc_tmp, lon_enc_tmp), 1)
                    fut_pred.append(self.decode(enc_tmp))
            return fut_pred, lat_pred, lon_pred


    def decode(self,enc):
        enc = enc.repeat(self.out_length, 1, 1)
        h_dec, _ = self.dec_lstm(enc)
        h_dec = h_dec.permute(1, 0, 2)
        fut_pred = self.op(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred = outputActivation(fut_pred)
        return fut_pred