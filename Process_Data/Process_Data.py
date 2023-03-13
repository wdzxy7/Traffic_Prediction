import os
import math
import h5py
import numpy as np


class MinMaxNormalization(object):
    '''MinMax Normalization --> [-1, 1]
       x = (x - min) / (max - min).
       x = x * 2 - 1
    '''

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max: ", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X


def deal_bj():
    print('Bj data process start')
    all_data = []
    all_time = []
    bj_save_path = os.path.join(save_path, 'TaxiBJ')
    if not os.path.exists(bj_save_path):
        os.makedirs(bj_save_path)
    for year in range(13, 17):
        fpath = os.path.join(data_path, 'TaxiBJ', 'BJ{}_M32x32_T30_InOut.h5'.format(year))
        print("read file name: ", fpath)
        flow_data, time_data = load_data(fpath)
        flow_data, time_data = clean_data(flow_data, time_data)
        flow_data = flow_data[:, :2]
        flow_data[flow_data < 0] = 0
        print(len(flow_data), time_data[0], time_data[-1])
        all_data.append(flow_data)
        all_time.append(time_data)
    all_data = np.vstack(all_data)
    print('all data shape', all_data.shape)
    # normal data
    mmn = MinMaxNormalization()
    mmn.fit(all_data)
    mmn_all_data = [mmn.transform(d) for d in all_data]
    all_time = np.concatenate(all_time)
    # spilt data
    len_train, len_val, len_test = split_data(all_data.shape[0], test_rate, val_rate)
    train_data = mmn_all_data[0: len_train]
    train_time = all_time[0: len_train]
    test_data = mmn_all_data[len_train: len_train + len_test]
    test_time = all_time[len_train: len_train + len_test]
    val_data = mmn_all_data[-len_val:]
    val_time = all_time[-len_val:]
    # save data
    np.save(os.path.join(bj_save_path, 'raw_all_data'), all_data)
    np.save(os.path.join(bj_save_path, 'all_data'), mmn_all_data)
    np.save(os.path.join(bj_save_path, 'all_time'), all_time)
    np.save(os.path.join(bj_save_path, 'train_data'), train_data)
    np.save(os.path.join(bj_save_path, 'train_time'), train_time)
    np.save(os.path.join(bj_save_path, 'test_data'), test_data)
    np.save(os.path.join(bj_save_path, 'test_time'), test_time)
    np.save(os.path.join(bj_save_path, 'val_data'), val_data)
    np.save(os.path.join(bj_save_path, 'val_time'), val_time)
    print('data process over')


def deal_bike_nyc():
    print('bike nyc data process start')
    bike_nyc_save_path = os.path.join(save_path, 'BikeNYC')
    if not os.path.exists(bike_nyc_save_path):
        os.makedirs(bike_nyc_save_path)
    fpath = os.path.join(data_path, 'BikeNYC', 'NYC14_M16x8_T60_NewEnd.h5')
    all_data, all_time = load_data(fpath)
    print('all data shape', all_data.shape)
    mmn = MinMaxNormalization()
    mmn.fit(all_data)
    mmn_all_data = [mmn.transform(d) for d in all_data]
    len_train, len_val, len_test = split_data(all_data.shape[0], test_rate, val_rate)
    train_data = mmn_all_data[0: len_train]
    train_time = all_time[0: len_train]
    test_data = mmn_all_data[len_train: len_train + len_test]
    test_time = all_time[len_train: len_train + len_test]
    val_data = mmn_all_data[-len_val:]
    val_time = all_time[-len_val:]
    np.save(os.path.join(bike_nyc_save_path, 'raw_all_data'), all_data)
    np.save(os.path.join(bike_nyc_save_path, 'all_data'), mmn_all_data)
    np.save(os.path.join(bike_nyc_save_path, 'all_time'), all_time)
    np.save(os.path.join(bike_nyc_save_path, 'train_data'), train_data)
    np.save(os.path.join(bike_nyc_save_path, 'train_time'), train_time)
    np.save(os.path.join(bike_nyc_save_path, 'test_data'), test_data)
    np.save(os.path.join(bike_nyc_save_path, 'test_time'), test_time)
    np.save(os.path.join(bike_nyc_save_path, 'val_data'), val_data)
    np.save(os.path.join(bike_nyc_save_path, 'val_time'), val_time)
    print('data process over')


def deal_taxi_nyc():
    print('taxi nyc data process start')
    taxi_nyc_save_path = os.path.join(save_path, 'TaxiNYC')
    if not os.path.exists(taxi_nyc_save_path):
        os.makedirs(taxi_nyc_save_path)
    fpath = os.path.join(data_path, 'TaxiNYC', 'NYC2014.h5')
    all_data, all_time = load_data(fpath)
    print('all data shape', all_data.shape)
    mmn = MinMaxNormalization()
    mmn.fit(all_data)
    mmn_all_data = [mmn.transform(d) for d in all_data]
    len_train, len_val, len_test = split_data(all_data.shape[0], test_rate, val_rate)
    train_data = mmn_all_data[0: len_train]
    train_time = all_time[0: len_train]
    test_data = mmn_all_data[len_train: len_train + len_test]
    test_time = all_time[len_train: len_train + len_test]
    val_data = mmn_all_data[-len_val:]
    val_time = all_time[-len_val:]
    np.save(os.path.join(taxi_nyc_save_path, 'raw_all_data'), all_data)
    np.save(os.path.join(taxi_nyc_save_path, 'all_data'), mmn_all_data)
    np.save(os.path.join(taxi_nyc_save_path, 'all_time'), all_time)
    np.save(os.path.join(taxi_nyc_save_path, 'train_data'), train_data)
    np.save(os.path.join(taxi_nyc_save_path, 'train_time'), train_time)
    np.save(os.path.join(taxi_nyc_save_path, 'test_data'), test_data)
    np.save(os.path.join(taxi_nyc_save_path, 'test_time'), test_time)
    np.save(os.path.join(taxi_nyc_save_path, 'val_data'), val_data)
    np.save(os.path.join(taxi_nyc_save_path, 'val_time'), val_time)
    print('data process over')


def split_data(len_alldata, test_percent, val_percent):
    len_train = math.ceil(len_alldata * (1 - val_percent - test_percent))
    len_val = int(len_alldata * val_percent)
    len_test = int(len_alldata * test_percent)
    return len_train, len_val, len_test


def load_data(fpath):
    # read data file
    with h5py.File(fpath) as f:
        flow_data = f['data'].value
        time_data = f['date'].value
    return flow_data, time_data


def clean_data(flow_data, time_data, T=48):
    # remove some incomplete day, Use ASTCN code
    complete_days = []
    i = 0
    data_lens = len(time_data)
    while i < data_lens:
        if int(time_data[i][8:]) != 1:
            i += 1
        elif i + T - 1 < data_lens and int(time_data[i + T - 1][8:]) == T:
            complete_days.append(time_data[i][:8])
            i += T
        else:
            i += 1
    complete_days = set(complete_days)
    idx = []
    for i, t in enumerate(time_data):
        if t[:8] in complete_days:
            idx.append(i)
    flow_data = flow_data[idx]
    time_data = [time_data[i] for i in idx]
    return flow_data, time_data


if __name__ == '__main__':
    test_rate = 0.2
    val_rate = 0.2
    data_path = '../Data'
    save_path = '../processed'
    deal_taxi_nyc()
