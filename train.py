import os
import time
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import pipNet
from data import highwayTrajDataset
from utils import initLogging, maskedNLL, maskedMSE, maskedNLLTest

## Network Arguments
parser = argparse.ArgumentParser(description='Training: Planning-informed Trajectory Prediction for Autonomous Driving')
# General setting------------------------------------------
parser.add_argument('--use_cuda', action='store_false', help='if use cuda (default: True)', default = True)
parser.add_argument('--use_planning', action="store_false", help='if use planning coupled module (default: True)',default = True)
parser.add_argument('--use_fusion', action="store_false", help='if use targets fusion module (default: True)',default = True)
parser.add_argument('--train_output_flag', action="store_false", help='if concatenate with true maneuver label (default: True)', default = True)
parser.add_argument('--batch_size', type=int, help='batch size to use (default: 64)',  default=64)
parser.add_argument('--learning_rate', type=float, help='learning rate (default: 1e-3)', default=0.001)
parser.add_argument('--tensorboard', action="store_true", help='if use tensorboard (default: True)', default = True)
# IO setting------------------------------------------
parser.add_argument('--grid_size', type=int,  help='default: (25,5)', nargs=2,    default = [25, 5])
parser.add_argument('--in_length', type=int,  help='History sequence (default: 16)',default = 16)    # 3s history traj at 5Hz
parser.add_argument('--out_length', type=int, help='Predict sequence (default: 25)',default = 25)    # 5s future traj at 5Hz
parser.add_argument('--num_lat_classes', type=int, help='Classes of lateral behaviors',     default = 3)
parser.add_argument('--num_lon_classes', type=int, help='Classes of longitute behaviors',   default = 2)
# Network hyperparameters------------------------------------------
parser.add_argument('--temporal_embedding_size', type=int,  help='Embedding size of the input traj', default = 32)
parser.add_argument('--encoder_size', type=int, help='lstm encoder size',  default = 64)
parser.add_argument('--decoder_size', type=int, help='lstm decoder size',  default = 128)
parser.add_argument('--soc_conv_depth', type=int, help='The 1st social conv depth',  default = 64)
parser.add_argument('--soc_conv2_depth', type=int, help='The 2nd social conv depth',  default = 16)
parser.add_argument('--dynamics_encoding_size', type=int,  help='Embedding size of the vehicle dynamic',  default = 32)
parser.add_argument('--social_context_size', type=int,  help='Embedding size of the social context tensor',  default = 80)
parser.add_argument('--fuse_enc_size', type=int,  help='Feature size to be fused',  default = 112)
# Training setting------------------------------------------
parser.add_argument('--name', type=str, help='log name (default: "1")', default="1")
parser.add_argument('--train_set', type=str, help='Path to train datasets')
parser.add_argument('--val_set', type=str, help='Path to validation datasets')
parser.add_argument("--num_workers", type=int, default=8, help="number of workers used for dataloader")
parser.add_argument('--pretrain_epochs', type=int, help='epochs of pre-training using MSE', default = 5)
parser.add_argument('--train_epochs',    type=int, help='epochs of training using NLL', default = 10)


def train_model():
    args = parser.parse_args()

    ## Logging
    log_path = "./trained_models/{}/".format(args.name)
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+'train.log')
    if args.tensorboard:
        logger = SummaryWriter(log_path + 'train-pre{}-nll{}'.format(args.pretrain_epochs, args.train_epochs))
        logger_val = SummaryWriter(log_path + 'validation-pre{}-nll{}'.format(args.pretrain_epochs, args.train_epochs))

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Batch size : {}".format(args.batch_size))
    logging.info("Learning rate : {}".format(args.learning_rate))
    logging.info("Use Planning Coupled: {}".format(args.use_planning))
    logging.info("Use Target Fusion: {}".format(args.use_fusion))

    ## Initialize network and optimizer
    PiP = pipNet(args)
    if args.use_cuda:
        PiP = PiP.cuda()
    optimizer = torch.optim.Adam(PiP.parameters(), lr=args.learning_rate)
    crossEnt = torch.nn.BCELoss()

    ## Initialize training parameters
    pretrainEpochs = args.pretrain_epochs
    trainEpochs    = args.train_epochs
    batch_size     = args.batch_size

    ## Initialize data loaders
    logging.info("Train dataset: {}".format(args.train_set))
    trSet = highwayTrajDataset(path=args.train_set,
                         targ_enc_size=args.social_context_size+args.dynamics_encoding_size,
                         grid_size=args.grid_size,
                         fit_plan_traj=False)
    logging.info("Validation dataset: {}".format(args.val_set))
    valSet = highwayTrajDataset(path=args.val_set,
                          targ_enc_size=args.social_context_size+args.dynamics_encoding_size,
                          grid_size=args.grid_size,
                          fit_plan_traj=True)
    trDataloader =  DataLoader(trSet, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=trSet.collate_fn)
    valDataloader = DataLoader(valSet, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=valSet.collate_fn)
    logging.info("DataSet Prepared : {} train data, {} validation data\n".format(len(trSet), len(valSet)))
    logging.info("Network structure: {}\n".format(PiP))

    ## Training process
    for epoch_num in range( pretrainEpochs + trainEpochs ):
        if epoch_num == 0:
            logging.info('Pretrain with MSE loss')
        elif epoch_num == pretrainEpochs:
            logging.info('Train with NLL loss')
        ## Variables to track training performance:
        avg_time_tr, avg_loss_tr, avg_loss_val = 0, 0, 0
        ## Training status, reclaim after each epoch
        PiP.train()
        PiP.train_output_flag = True
        for i, data in enumerate(trDataloader):
            st_time = time.time()
            nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, targsFut, targsFutMask, lat_enc, lon_enc, _ = data
            if args.use_cuda:
                nbsHist = nbsHist.cuda()
                nbsMask = nbsMask.cuda()
                planFut = planFut.cuda()
                planMask = planMask.cuda()
                targsHist = targsHist.cuda()
                targsEncMask = targsEncMask.cuda()
                lat_enc = lat_enc.cuda()
                lon_enc = lon_enc.cuda()
                targsFut = targsFut.cuda()
                targsFutMask = targsFutMask.cuda()

            # Forward pass
            fut_pred, lat_pred, lon_pred = PiP(nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, lat_enc, lon_enc)
            if epoch_num < pretrainEpochs:
                # Pre-train with MSE loss to speed up training
                l = maskedMSE(fut_pred, targsFut, targsFutMask)
            else:
                # Train with NLL loss
                l = maskedNLL(fut_pred, targsFut, targsFutMask) + crossEnt(lat_pred, lat_enc) + crossEnt(lon_pred, lon_enc)

            # Back-prop and update weights
            optimizer.zero_grad()
            l.backward()
            prev_vec_norm = torch.nn.utils.clip_grad_norm_(PiP.parameters(), 10)
            optimizer.step()

            # Track average train loss and average train time:
            batch_time = time.time()-st_time
            avg_loss_tr += l.item()
            avg_time_tr += batch_time

            # For every 100 batches: record loss, validate model, and plot.
            if i%100 == 99:
                eta = avg_time_tr/100*(len(trSet)/batch_size-i)
                epoch_progress = i * batch_size / len(trSet)
                logging.info(f"Epoch no:{epoch_num+1}"+
                             f" | Epoch progress(%):{epoch_progress*100:.2f}"+
                             f" | Avg train loss:{avg_loss_tr/100:.2f}"+
                             f" | ETA(s):{int(eta)}")

                if args.tensorboard:
                    logger.add_scalar("RMSE" if epoch_num < pretrainEpochs else "NLL", avg_loss_tr / 100, (epoch_progress + epoch_num) * 100)

                ## Validatation during training:
                eval_batch_num = 20
                with torch.no_grad():
                    PiP.eval()
                    PiP.train_output_flag = False
                    for i, data in enumerate(valDataloader):
                        nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, targsFut, targsFutMask, lat_enc, lon_enc, _ = data
                        if args.use_cuda:
                            nbsHist = nbsHist.cuda()
                            nbsMask = nbsMask.cuda()
                            planFut = planFut.cuda()
                            planMask = planMask.cuda()
                            targsHist = targsHist.cuda()
                            targsEncMask = targsEncMask.cuda()
                            lat_enc = lat_enc.cuda()
                            lon_enc = lon_enc.cuda()
                            targsFut = targsFut.cuda()
                            targsFutMask = targsFutMask.cuda()
                        if epoch_num < pretrainEpochs:
                            # During pre-training with MSE loss, validate with MSE for true maneuver class trajectory
                            PiP.train_output_flag = True
                            fut_pred, _, _ = PiP(nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask,
                                                 lat_enc, lon_enc)
                            l = maskedMSE(fut_pred, targsFut, targsFutMask)
                        else:
                            # During training with NLL loss, validate with NLL over multi-modal distribution
                            fut_pred, lat_pred, lon_pred = PiP(nbsHist, nbsMask, planFut, planMask, targsHist,
                                                               targsEncMask, lat_enc, lon_enc)
                            l = maskedNLLTest(fut_pred, lat_pred, lon_pred, targsFut, targsFutMask, avg_along_time=True)
                        avg_loss_val += l.item()
                        if i==(eval_batch_num-1):
                            if args.tensorboard:
                                logger_val.add_scalar("RMSE" if epoch_num < pretrainEpochs else "NLL", avg_loss_val / eval_batch_num, (epoch_progress + epoch_num) * 100)
                            break
                # Clear statistic
                avg_time_tr, avg_loss_tr, avg_loss_val = 0, 0, 0
                # Revert to train mode after in-process evaluation.
                PiP.train()
                PiP.train_output_flag = True

        ## Save the model after each epoch______________________________________________________________________________
        epoCount = epoch_num + 1
        if epoCount < pretrainEpochs:
            torch.save(PiP.state_dict(), log_path + "{}-pre{}-nll{}.tar".format(args.name, epoCount, 0))
        else:
            torch.save(PiP.state_dict(), log_path + "{}-pre{}-nll{}.tar".format(args.name, pretrainEpochs, epoCount - pretrainEpochs))

    # All epochs finish________________________________________________________________________________________________
    torch.save(PiP.state_dict(), log_path+"{}.tar".format(args.name))
    logging.info("Model saved in trained_models/{}/{}.tar\n".format(args.name, args.name))

if __name__ == '__main__':
    train_model()