import os
import h5py
import time
import torch
import numpy as np
import torch.utils.data as data


class FlowDataset(data.Dataset):
    def __init__(self, data_name, window_size=7 * 48, data_type='train', use_ext=False):
        self.raw_data_path = './Data'
        self.data_path = './processed'
        self.data_name = data_name
        self.wind_size = window_size
        self.data_type = data_type
        self.holidays = None
        self.flow_data = None
        self.time_data = None
        self.data_len = 0
        self.use_ext = use_ext
        if data_name != 'TaxiCQ':
            self.meteorol_data = self.get_all_meteorol()
        self.load_data()

    def __getitem__(self, index):
        end_index = index + self.wind_size
        x = self.flow_data[index: end_index]
        y = self.flow_data[end_index: end_index + 1]
        time_data = self.time_data[end_index - 4: end_index]
        external_data = self.load_external(time_data)
        external_data = torch.tensor(external_data).long()
        # move channel to first
        return x.permute(1, 0, 2, 3), y.permute(1, 0, 2, 3), external_data

    def __len__(self):
        return self.data_len

    def load_external(self, time_data):
        if self.use_ext is False:
            return np.asarray([])
        vec_time = self.timestamp2vec(time_data)
        holiday_data = self.load_holiday([str(int(x)) for x in time_data])
        if self.data_name != 'TaxiCQ':
            meteorol = self.load_meteorol(time_data)
            return np.hstack([vec_time, holiday_data, meteorol])
        else:
            return np.hstack([vec_time, holiday_data])

    # copy from astcn
    def timestamp2vec(self, timestamps):
        vec = [time.strptime(str(t[:8], encoding='utf-8'), '%Y%m%d').tm_wday for t in timestamps]
        ret = []
        for i in vec:
            v = [0 for _ in range(7)]
            v[i] = 1
            if i >= 5:
                v.append(0)
            else:
                v.append(1)
            ret.append(v)
        return np.asarray(ret)

    # copy from astcn
    def load_holiday(self, timeslots):
        holidays = self.holidays
        holidays = set([h.strip() for h in holidays])
        H = np.zeros(len(timeslots))
        for i, slot in enumerate(timeslots):
            if slot[:8] in holidays:
                H[i] = 1
        return H[:, None]

    def get_all_meteorol(self):
        fpath = os.path.join(self.raw_data_path, self.data_name, 'Meteorology.h5')
        meteorol_file = h5py.File(fpath, 'r')
        meteoroal = {}
        for key in meteorol_file.keys():
            meteoroal[key] = meteorol_file[key][()]
        return meteoroal

    def load_meteorol(self, timeslots):
        '''
        timeslots: the predicted timeslots
        In real-world, we dont have the meteorol data in the predicted timeslot, instead, we use the meteoral at previous timeslots, i.e., slot = predicted_slot - timeslot (you can use predicted meteorol data as well)
        '''
        f = self.meteorol_data
        Timeslot = f['date']
        WindSpeed = f['WindSpeed']
        Weather = f['Weather']
        Temperature = f['Temperature']

        M = dict()  # map timeslot to index
        for i, slot in enumerate(Timeslot):
            M[slot] = i

        WS = []  # WindSpeed
        WR = []  # Weather
        TE = []  # Temperature
        for slot in timeslots:
            predicted_id = M[slot]
            cur_id = predicted_id - 1
            WS.append(WindSpeed[cur_id])
            WR.append(Weather[cur_id])
            TE.append(Temperature[cur_id])

        WS = np.asarray(WS)
        WR = np.asarray(WR)
        TE = np.asarray(TE)
        # 0-1 scale
        WS = 1. * (WS - WS.min()) / (WS.max() - WS.min())
        TE = 1. * (TE - TE.min()) / (TE.max() - TE.min())
        merge_data = np.hstack([WR, WS[:, None], TE[:, None]])
        return merge_data

    def load_data(self):
        # load flow data
        data_path = os.path.join(self.data_path, self.data_name)
        if self.data_type == 'train':
            self.flow_data = np.load(os.path.join(data_path, 'train_data.npy'))
            self.time_data = np.load(os.path.join(data_path, 'train_time.npy'))
        elif self.data_type == 'test':
            self.flow_data = np.load(os.path.join(data_path, 'test_data.npy'))
            self.time_data = np.load(os.path.join(data_path, 'test_time.npy'))
        elif self.data_type == 'val':
            self.flow_data = np.load(os.path.join(data_path, 'val_data.npy'))
            self.time_data = np.load(os.path.join(data_path, 'val_time.npy'))
        else:
            self.flow_data = np.load(os.path.join(data_path, 'all_data.npy'))
            self.time_data = np.load(os.path.join(data_path, 'all_time.npy'))
        # the last 7 day is incomplete
        self.data_len = len(self.flow_data) - self.wind_size
        self.flow_data = torch.tensor(self.flow_data).float()
        # load holiday data
        if self.data_name == 'TaxiBJ':
            fname = os.path.join(self.raw_data_path, self.data_name, 'BJ_Holiday.txt')
            with open(fname, 'r') as f:
                self.holidays = f.readlines()
        elif self.data_name == 'BikeNYC':
            fname = os.path.join(self.raw_data_path, self.data_name, 'NY_Holiday.txt')
            with open(fname, 'r') as f:
                self.holidays = f.readlines()
        elif self.data_name == 'TaxiNYC':
            fname = os.path.join(self.raw_data_path, self.data_name, 'Holiday.txt')
            with open(fname, 'r') as f:
                self.holidays = f.readlines()
        elif self.data_name == 'TaxiCQ':
            fname = os.path.join(self.raw_data_path, self.data_name, 'CQ_Holiday.txt')
            with open(fname, 'r') as f:
                self.holidays = f.readlines()
        elif self.data_name == 'PoolTaxiBJ':
            fname = os.path.join(self.raw_data_path, self.data_name, 'BJ_Holiday.txt')
            with open(fname, 'r') as f:
                self.holidays = f.readlines()