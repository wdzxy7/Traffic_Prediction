import os
import torch
import time
import numpy as np
import torch.utils.data as data


class FlowDataset(data.Dataset):
    def __init__(self, data_name, window_size=7 * 48, data_type='train'):
        self.raw_data_path = './Data'
        self.data_path = './processed'
        self.data_name = data_name
        self.wind_size = window_size
        self.data_type = data_type
        self.holidays = None
        self.flow_data = None
        self.time_data = None
        self.data_len = 0
        self.load_data()

    def __getitem__(self, index):
        end_index = index + self.wind_size
        x = self.flow_data[index: end_index]
        y = self.flow_data[end_index: end_index + 1]
        time_data = self.time_data[index: end_index]
        external_data = self.load_external(time_data)
        external_data = torch.tensor(external_data).float()
        # move channel to first
        return x.permute(1, 0, 2, 3), y.permute(1, 0, 2, 3), external_data

    def __len__(self):
        return self.data_len

    def load_external(self, time_data):
        vec_time = self.timestamp2vec(time_data)
        holiday_data = self.load_holiday([str(int(x)) for x in time_data])
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