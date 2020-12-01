import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import quintic_spline, fitting_traj_by_qs

## NGSIM / HighD datasets are publicly available datasets
## First using preprocess code to make them into mat file with the following format:

'''
% Data: #row = data number, #column = 138 (13+grid_num)
%{
0: Dataset Id
1: Vehicle Id
|2 : Frame Id
|3 : Local X
|4 : Local Y
|5 : Lane Id
|6 : Lateral maneuver
|7 : Longitudinal maneuver
|8 : Length
|9 : Width
|10: Class label
|11: Velocity
|12: Accerlation
|13-137: Neighbor Car Ids at grid location
%}
'''

'''
% Tracks: cells: {Dataset_Id * Vehicle_Id}, each cell: #row = 136 (11+grid_num), #column=totalFramNum
%{
|0 : Frame Id
|1 : Local X
|2 : Local Y
|3 : Lane Id
|4 : Lateral maneuver
|5 : Longitudinal maneuver
|6 : Length
|7 : Width
|8 : Class label
|9 : Velocity
|10: Accerlation
|11-135: Neighbor Car Ids at grid location
'''

### Class for the highway trajectory datasets (NISIM, HighD, etc.)
class highwayTrajDataset(Dataset):

    def __init__(self, path, t_h=30, t_f=50, d_s=2,
                 enc_size=64, targ_enc_size=112, grid_size=(25, 5), fit_plan_traj=False, fit_plan_further_ds=1):
        if not os.path.exists(path):
            raise RuntimeError("{} not exists!!".format(path))
        if path.endswith('.mat'):
            f = h5py.File(path, 'r')
            f_tracks = f['tracks']
            track_cols, track_rows = f_tracks.shape
            self.Data = np.transpose(f['traj'])
            self.Tracks = []
            for i in range(track_rows):
                self.Tracks.append([  np.transpose(f[f_tracks[j][i]][:]) for j in range(track_cols)  ])
        else:
            raise RuntimeError("Path should be end with '.mat' for file or '/' for folder")

        # If torch version >= 1.2.0
        if int(torch.__version__[0])>=1 and int(torch.__version__[2])>=2:
            self.mask_num_type = torch.bool
        else:
            self.mask_num_type = torch.uint8

        self.t_h = t_h  # length of track history.
        self.t_f = t_f  # length of predicted trajectory.
        self.d_s = d_s  # downsampling rate of all trajectories to be processed.
        self.enc_size = enc_size
        self.targ_enc_size = targ_enc_size
        self.hist_len = self.t_h // self.d_s + 1  # data length of the history trajectory
        self.fut_len = self.t_f // self.d_s       # data length of the future  trajectory
        self.plan_len = self.t_f // self.d_s      # data length of the planning  trajectory

        self.fit_plan_traj = fit_plan_traj              # Fitting the future planned trajectory in testing/evaluation.
        self.further_ds_plan = fit_plan_further_ds      # Further downsampling to restrict the planning info

        self.cell_length = 8
        self.cell_width = 7
        self.grid_size = grid_size                # size of social context grid
        self.grid_cells = grid_size[0] * grid_size[1]
        self.grid_length = self.cell_length * grid_size[0]
        self.grid_width = self.cell_width * grid_size[1]

    def __len__(self):
        return len(self.Data)

    ## Functions of retrieving information according to the item's index
    def itsDsId(self, idx):
        return self.Data[idx, 0].astype(int)

    def itsPlanVehId(self, idx):
        return self.Data[idx, 1].astype(int)

    def itsTime(self, idx):
        return self.Data[idx, 2]

    def itsLocation(self, idx):
        return self.Data[idx, 3:5]

    def itsPlanVehBehavior(self, idx):
        return int(self.Data[idx, 6] + (self.Data[idx, 7] - 1) * 3)

    def itsPlanVehSize(self, idx):
        return self.Data[idx, 8:10]

    def itsPlanVehDynamic(self, idx):
        planVel, planAcc = self.getDynamic(self.itsDsId(idx), self.itsPlanVehId(idx), self.itsTime(idx))
        return planVel, planAcc

    def itsCentGrid(self, idx):
        return self.Data[idx, 13:].astype(int)

    def itsTargVehsId(self, idx):
        centGrid = self.itsCentGrid(idx)
        targVehsId = centGrid[np.nonzero(centGrid)]
        return targVehsId

    def itsNbrVehsId(self, idx):
        dsId = self.itsDsId(idx)
        planVehId = self.itsPlanVehId(idx)
        targVehsId = self.itsTargVehsId(idx)
        t = self.itsTime(idx)
        nbrVehsId = np.array([], dtype=np.int64)
        for target in targVehsId:
            subGrid = self.getGrid(dsId, target, t)
            subIds = subGrid[np.nonzero(subGrid)]
            for i in subIds:
                if i==planVehId or any(i==targVehsId) or any(i==nbrVehsId):
                    continue
                else:
                    nbrVehsId = np.append(nbrVehsId, i)
        return nbrVehsId

    def itsTargsCentLoc(self, idx):
        dsId = self.itsDsId(idx)
        t = self.itsTime(idx)
        centGrid = self.itsCentGrid(idx)
        targsCenterLoc = np.empty((0,2), dtype=np.float32)
        for target in centGrid:
            if target:
                targsCenterLoc = np.vstack([targsCenterLoc, self.getLocation(dsId, target, t)])
        return torch.from_numpy(targsCenterLoc)

    def itsAllAroundSizes(self, idx):
        dsId = self.itsDsId(idx)
        centGrid = self.itsCentGrid(idx)
        t = self.itsTime(idx)
        planVehSize = []
        targVehSizes = []
        nbsVehSizes = []
        planVehSize.append(self.getSize(dsId, self.itsPlanVehId(idx)))
        for i, target in enumerate(centGrid):
            if target:
                targVehSizes.append(self.getSize(dsId, target))
                targVehGrid = self.getGrid(dsId, target, t)
                for targetNb in targVehGrid:
                    if targetNb:
                        nbsVehSizes.append(self.getSize(dsId, targetNb))
        return np.asarray(planVehSize), np.asarray(targVehSizes), np.asarray(nbsVehSizes)

    ## Functions for retrieving trajectory data with absolute coordinate, mainly used for visualization
    def itsAllGroundTruthTrajs(self, idx):
        return [self.absPlanTraj(idx), self.absTargsTraj(idx), self.absNbrsTraj(idx)]

    def absPlanTraj(self, idx):
        dsId = self.itsDsId(idx)
        planVeh = self.itsPlanVehId(idx)
        t = self.itsTime(idx)
        colIndex = np.where(self.Tracks[dsId - 1][planVeh - 1][0, :] == t)[0][0]
        vehTrack = self.Tracks[dsId - 1][planVeh - 1].transpose()
        planHis = vehTrack[np.maximum(0, colIndex - self.t_h): (colIndex + 1): self.d_s, 1:3]
        planFut = vehTrack[(colIndex + self.d_s): (colIndex + self.t_f + 1): self.d_s, 1:3]
        return [planHis, planFut]

    def absTargsTraj(self, idx):
        dsId = self.itsDsId(idx)
        targVehs = self.itsTargVehsId(idx)
        t = self.itsTime(idx)
        targHisList, targFutList = [], []
        for targVeh in targVehs:
            colIndex = np.where(self.Tracks[dsId - 1][targVeh - 1][0, :] == t)[0][0]
            vehTrack = self.Tracks[dsId - 1][targVeh - 1].transpose()
            targHis = vehTrack[np.maximum(0, colIndex - self.t_h): (colIndex + 1): self.d_s, 1:3]
            targFut = vehTrack[(colIndex + self.d_s): (colIndex + self.t_f + 1): self.d_s, 1:3]
            targHisList.append(targHis)
            targFutList.append(targFut)
        return [targHisList, targFutList]

    def absNbrsTraj(self, idx):
        dsId = self.itsDsId(idx)
        nbrVehs = self.itsNbrVehsId(idx)
        t = self.itsTime(idx)
        nbrHisList, nbrFutList = [], []
        for nbrVeh in nbrVehs:
            colIndex = np.where(self.Tracks[dsId - 1][nbrVeh - 1][0, :] == t)[0][0]
            vehTrack = self.Tracks[dsId - 1][nbrVeh - 1].transpose()
            targHis = vehTrack[np.maximum(0, colIndex - self.t_h): (colIndex + 1): self.d_s, 1:3]
            nbrHisList.append(targHis)
        return [nbrHisList, nbrFutList]

    def batchTargetVehsInfo(self, idxs):
        count = 0
        dsIds = np.zeros(len(idxs)*self.grid_cells, dtype=int)
        vehIds = np.zeros(len(idxs)*self.grid_cells, dtype=int)
        for idx in idxs:
            dsId = self.itsDsId(idx)
            targets = self.itsCentGrid(idx)
            targetsIndex = np.nonzero(targets)
            for index in targetsIndex[0]:
                dsIds[count] = dsId
                vehIds[count] = targets[index]
                count += 1
        return [dsIds[:count], vehIds[:count]]

    ## Avoid searching the correspond column for too many times.
    def getTracksCol(self, dsId, vehId, t):
        return np.where(self.Tracks[dsId - 1][vehId - 1][0, :] == t)[0][0]

    ## Get the vehicle's location from tracks
    def getLocation(self, dsId, vehId, t):
        colIndex = np.where(self.Tracks[dsId - 1][vehId - 1][0, :] == t)[0][0]
        location = self.getLocationByCol(dsId, vehId, colIndex)
        return location
    def getLocationByCol(self, dsId, vehId, colIndex):
        return self.Tracks[dsId - 1][vehId - 1][1:3, colIndex].transpose()

    ## Get the vehicle's maneuver given dataset id, vehicle id and time point t.
    def getManeuver(self, dsId, vehId, t):
        colIndex = np.where(self.Tracks[dsId - 1][vehId - 1][0, :] == t)[0][0]
        lat_lon_maneuvers = self.getManeuverByCol(dsId, vehId, colIndex)
        return lat_lon_maneuvers
    def getManeuverByCol(self, dsId, vehId, colIndex):
        return self.Tracks[dsId - 1][vehId - 1][4:6, colIndex].astype(int)

    ## Get the vehicle's nearby neighbours
    def getGrid(self, dsId, vehId, t):
        colIndex = np.where(self.Tracks[dsId - 1][vehId - 1][0, :] == t)[0][0]
        grid = self.getGridByCol(dsId, vehId, colIndex)
        return grid
    def getGridByCol(self, dsId, vehId, colIndex):
        return self.Tracks[dsId - 1][vehId - 1][11:, colIndex].astype(int)

    ## Get the vehicle's dynamic (velocity & acceleration) given dataset id, vehicle id and time point t.
    def getDynamic(self, dsId, vehId, t):
        colIndex = np.where(self.Tracks[dsId - 1][vehId - 1][0, :] == t)[0][0]
        vel_acc = self.getDynamicByCol(dsId, vehId, colIndex)
        return vel_acc
    def getDynamicByCol(self, dsId, vehId, colIndex):
        return self.Tracks[dsId - 1][vehId - 1][9:11, colIndex]

    ## Get the vehicle's size (length & width) given dataset id and vehicle id
    def getSize(self, dsId, vehId):
        length_width = self.Tracks[dsId - 1][vehId - 1][6:8, 0]
        return length_width

    ## Helper function to get track history
    def getHistory(self, dsId, vehId, refVehId, t, wholePeriod=False):
        if vehId == 0:
            # if return empty, it denotes there's no vehicle in that grid.
            return np.empty([0, 2])
        else:
            vehColIndex = np.where(self.Tracks[dsId - 1][vehId - 1][0, :] == t)[0][0]
            refColIndex = np.where(self.Tracks[dsId - 1][refVehId - 1][0, :] == t)[0][0]
            vehTrack = self.Tracks[dsId - 1][vehId - 1][1:3].transpose()
            refTrack = self.Tracks[dsId - 1][refVehId - 1][1:3].transpose()
            # Use the sequence of trajectory or just the last instance as the refPos
            if wholePeriod:
                refStpt = np.maximum(0, refColIndex - self.t_h)
                refEnpt = refColIndex + 1
                refPos = refTrack[refStpt:refEnpt:self.d_s, :]
            else:
                refPos = np.tile(refTrack[refColIndex, :], (self.hist_len, 1))
            stpt = np.maximum(0, vehColIndex - self.t_h)
            enpt = vehColIndex + 1
            vehPos = vehTrack[stpt:enpt:self.d_s, :]
            if len(vehPos) < self.hist_len:
                histPart = vehPos - refPos[-len(vehPos)::]
                paddingPart = np.tile(histPart[0, :], (self.hist_len - len(vehPos), 1))
                hist = np.concatenate((paddingPart, histPart), axis=0)
                return hist
            else:
                hist = vehPos - refPos
                return hist

    ## Helper function to get track future
    def getFuture(self, dsId, vehId, t):
        colIndex = np.where(self.Tracks[dsId - 1][vehId - 1][0, :] == t)[0][0]
        futTraj = self.getFutureByCol(dsId, vehId, colIndex)
        return futTraj
    def getFutureByCol(self, dsId, vehId, colIndex):
        vehTrack = self.Tracks[dsId - 1][vehId - 1].transpose()
        refPos = self.Tracks[dsId - 1][vehId - 1][1:3, colIndex].transpose()
        stpt = colIndex + self.d_s
        enpt = np.minimum(len(vehTrack), colIndex + self.t_f + 1)
        futTraj = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
        return futTraj

    def getPlanFuture(self, dsId, planId, refVehId, t):
        # Traj of the reference veh
        refColIndex = np.where(self.Tracks[dsId - 1][refVehId - 1][0, :] == t)[0][0]
        refPos = self.Tracks[dsId - 1][refVehId - 1][1:3, refColIndex].transpose()
        # Traj of the planned veh
        planColIndex = np.where(self.Tracks[dsId - 1][planId - 1][0, :] == t)[0][0]
        stpt = planColIndex
        enpt = planColIndex + self.t_f + 1
        planGroundTrue = self.Tracks[dsId - 1][planId - 1][1:3, stpt:enpt:self.d_s].transpose()
        planFut = planGroundTrue.copy()
        # Fitting the downsampled waypoints as the planned trajectory in testing and evaluation.
        if self.fit_plan_traj:
            wayPoint        = np.arange(0, self.t_f + self.d_s, self.d_s)
            wayPoint_to_fit = np.arange(0, self.t_f + 1, self.d_s * self.further_ds_plan)
            planFut_to_fit = planFut[::self.further_ds_plan, ]
            laterParam = fitting_traj_by_qs(wayPoint_to_fit, planFut_to_fit[:, 0])
            longiParam = fitting_traj_by_qs(wayPoint_to_fit, planFut_to_fit[:, 1])
            planFut[:, 0] = quintic_spline(wayPoint, *laterParam)
            planFut[:, 1] = quintic_spline(wayPoint, *longiParam)
        revPlanFut = np.flip(planFut[1:,] - refPos, axis=0).copy()
        return revPlanFut

    def __getitem__(self, idx):
        dsId = self.itsDsId(idx)
        centVehId = self.itsPlanVehId(idx)
        t = self.itsTime(idx)
        centGrid = self.itsCentGrid(idx)
        planGridLocs = []
        targsHists = []
        targsFuts = []
        targsLonEnc = []
        targsLatEnc = []
        nbsHists = []
        planFuts = []
        targsVehs = np.zeros(self.grid_cells)
        for id, target in enumerate(centGrid):
            if target:
                targetColumn = self.getTracksCol(dsId, target, t)
                # Get the grid of each neighbour vehicle.
                grid = self.getGridByCol(dsId, target, targetColumn)
                # Targets history and future
                targsVehs[id] = target
                targsHists.append(self.getHistory(dsId, target, target, t))
                targsFuts.append(self.getFutureByCol(dsId, target, targetColumn))
                # Targets maneuvers
                latMan, lonMan = self.getManeuverByCol(dsId, target, targetColumn)
                lat_enc = np.zeros([3])
                lon_enc = np.zeros([2])
                lat_enc[latMan - 1] = 1
                lon_enc[lonMan - 1] = 1
                targsLatEnc.append(lat_enc)
                targsLonEnc.append(lon_enc)
                # Neighbours history
                nbsHists.append([self.getHistory(dsId, i, target, t, wholePeriod=True) for i in grid])
                # PlanVeh future
                planGridLocs.append(np.where(grid == centVehId)[0][0])
                planFuts.append(self.getPlanFuture(dsId, centVehId, target, t))
        return planFuts, nbsHists, \
               targsHists, targsFuts, targsLonEnc, targsLatEnc, \
               centGrid, planGridLocs, idx


    ## Collate function for dataloader
    def collate_fn(self, samples):
        targs_batch_size = 0
        nbs_batch_size = 0
        for _, nbsHists, targsHists, _, _, _, _, _, _ in samples:
            targs_batch_size += len(targsHists)
            nbs_number = [sum([len(nbs) > 0 for nbs in sub_nbsHist]) for sub_nbsHist in nbsHists]
            nbs_batch_size += sum(nbs_number)
        # Initialize all things
        nbsHist_batch   = torch.zeros(self.hist_len,  nbs_batch_size,   2)
        targsHist_batch = torch.zeros(self.hist_len,  targs_batch_size, 2)
        targsFut_batch  = torch.zeros(self.fut_len,   targs_batch_size, 2)
        lat_enc_batch   = torch.zeros(targs_batch_size, 3)
        lon_enc_batch   = torch.zeros(targs_batch_size, 2)
        planFut_batch   = torch.zeros(self.plan_len,   targs_batch_size, 2)
        idxs = []
        pos = [0, 0]
        # Fill 1 on those grid locations with neighbour
        nbsMask_batch      = torch.zeros(targs_batch_size, self.grid_size[1], self.grid_size[0], self.enc_size, dtype=self.mask_num_type)
        planMask_batch     = torch.zeros(targs_batch_size, self.grid_size[1], self.grid_size[0], self.enc_size, dtype=self.mask_num_type)
        targsEncMask_batch = torch.zeros(len(samples),     self.grid_size[1], self.grid_size[0], self.targ_enc_size, dtype=self.mask_num_type)
        targsFutMask_batch = torch.zeros(self.fut_len,     targs_batch_size,  2)
        targetCount = 0
        nbCount = 0
        for i, (planFuts, nbsHists, targsHists, targsFuts, targsLonEnc, targsLatEnc, centGrid, planGridLocs, idx) in enumerate(samples):
            idxs.append(idx)
            centGridIndex = centGrid.nonzero()[0]
            for j in range(len(targsFuts)):
                targsHist_batch[0:len(targsHists[j]), targetCount, 0] = torch.from_numpy(targsHists[j][:, 0])
                targsHist_batch[0:len(targsHists[j]), targetCount, 1] = torch.from_numpy(targsHists[j][:, 1])
                targsFut_batch[0:len(targsFuts[j]), targetCount, 0] = torch.from_numpy(targsFuts[j][:, 0])
                targsFut_batch[0:len(targsFuts[j]), targetCount, 1] = torch.from_numpy(targsFuts[j][:, 1])
                targsFutMask_batch[0:len(targsFuts[j]), targetCount, :] = 1
                pos[0] = centGridIndex[j] % self.grid_size[0]
                pos[1] = centGridIndex[j] // self.grid_size[0]
                targsEncMask_batch[i, pos[1], pos[0], :] = torch.ones(self.targ_enc_size).byte()
                lat_enc_batch[targetCount, :] = torch.from_numpy(targsLatEnc[j])
                lon_enc_batch[targetCount, :] = torch.from_numpy(targsLonEnc[j])
                planFut_batch[0:len(planFuts[j]), targetCount, 0] = torch.from_numpy(planFuts[j][:, 0])
                planFut_batch[0:len(planFuts[j]), targetCount, 1] = torch.from_numpy(planFuts[j][:, 1])
                # Set up neighbor, neighbor sequence length, and mask batches:
                for index, nbHist in enumerate(nbsHists[j]):
                    if len(nbHist) != 0:
                        nbsHist_batch[0:len(nbHist), nbCount, 0] = torch.from_numpy(nbHist[:, 0])
                        nbsHist_batch[0:len(nbHist), nbCount, 1] = torch.from_numpy(nbHist[:, 1])
                        pos[0] = index % self.grid_size[0]
                        pos[1] = index // self.grid_size[0]
                        nbsMask_batch[targetCount, pos[1], pos[0], :] = torch.ones(self.enc_size).byte()
                        nbCount += 1
                        if index == planGridLocs[j]:
                            planMask_batch[targetCount, pos[1], pos[0], :] = torch.ones(self.enc_size).byte()
                targetCount += 1
        return nbsHist_batch, nbsMask_batch, \
               planFut_batch, planMask_batch, \
               targsHist_batch, targsEncMask_batch, \
               targsFut_batch, targsFutMask_batch, lat_enc_batch, lon_enc_batch, idxs
    # _______________________________________________________________________________________________________________________________________
