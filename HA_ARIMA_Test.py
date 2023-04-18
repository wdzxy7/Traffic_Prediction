import os
import h5py
import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.arima_model import ARIMA


def load_data(fpath):
    # read data file
    with h5py.File(fpath, 'r') as f:
        flow_data = f['data'][()]
        time_data = f['date'][()]
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


def get_past(data):
    T = 48

    past_data = []
    for i in range(1, 7):
        past_data.append(data[1 + i * T: 2 + i * T])

    # past_data = data
    past_data = np.vstack(past_data)
    return np.mean(past_data, axis=0)


def ha_test():
    if data_name == 'TaxiCQ':
        fpath = os.path.join(data_path, 'TaxiCQ', 'TaxiCQ_grid.h5')
        all_data, all_time = load_data(fpath)
        all_data, all_time = clean_data(all_data, all_time)
    elif data_name == 'TaxiBJ':
        all_data = []
        all_time = []
        for year in range(13, 17):
            fpath = os.path.join(data_path, 'TaxiBJ', 'BJ{}_M32x32_T30_InOut.h5'.format(year))
            flow_data, time_data = load_data(fpath)
            flow_data, time_data = clean_data(flow_data, time_data)
            flow_data = flow_data[:, :2]
            flow_data[flow_data < 0] = 0
            all_data.append(flow_data)
            all_time.append(time_data)
        all_data = np.vstack(all_data)
    elif data_name == 'BikeNYC':
        fpath = os.path.join(data_path, 'TaxiNYC', 'NYC2014.h5')
        all_data, all_time = load_data(fpath)
        all_data, all_time = clean_data(all_data, all_time)
    len_test = int(len(all_data) * 0.1)
    data = all_data[-len_test:]
    front = 0
    back = 336
    rmse_total = []
    mae_total = []
    c = 0
    while back < len_test:
        predict = get_past(data[front:back])
        true = data[back:back+1]
        rmse = (predict - true) ** 2
        mae = abs(predict - true)
        rmse_total.append(np.average(rmse))
        mae_total.append(np.average(mae))
        front += 1
        back += 1
        c += 1
    RMSE = np.sqrt(np.array(rmse_total).sum() / c)
    MAE = np.array(mae_total).sum() / c
    print(RMSE, MAE)


def arima_test():
    if data_name == 'TaxiCQ':
        fpath = os.path.join(data_path, 'TaxiCQ', 'TaxiCQ_grid.h5')
        all_data, all_time = load_data(fpath)
        all_data, all_time = clean_data(all_data, all_time)
        hs = 20
        ws = 25
    elif data_name == 'TaxiBJ':
        all_data = []
        all_time = []
        for year in range(13, 17):
            fpath = os.path.join(data_path, 'TaxiBJ', 'BJ{}_M32x32_T30_InOut.h5'.format(year))
            flow_data, time_data = load_data(fpath)
            flow_data, time_data = clean_data(flow_data, time_data)
            flow_data = flow_data[:, :2]
            flow_data[flow_data < 0] = 0
            all_data.append(flow_data)
            all_time.append(time_data)
        all_data = np.vstack(all_data)
        hs = 32
        ws = 32
    elif data_name == 'BikeNYC':
        fpath = os.path.join(data_path, 'TaxiNYC', 'NYC2014.h5')
        all_data, all_time = load_data(fpath)
        all_data, all_time = clean_data(all_data, all_time)
        hs = 8
        ws = 16
    len_test = 48
    data = all_data[-len_test:]
    data = data.transpose(1, 2, 3, 0)
    total_rmse = []
    total_mae = []
    for i in range(2):
        for h in range(hs):
            for w in range(ws):
                slice_data = data[i][h][w][:-2]
                true = data[i][h][w][-1:]
                model = pm.auto_arima(slice_data, start_p=1, start_q=1,
                                      information_criterion='aic',
                                      test='adf',  # use adftest to find optimal 'd'
                                      m=1,  # frequency of series
                                      d=None,  # let model determine 'd'
                                      seasonal=False,  # No Seasonality
                                      start_P=0,
                                      D=0,
                                      error_action='ignore',
                                      suppress_warnings=True)
                predict = model.predict(n_periods=1, return_conf_int=False)
                rmse = (predict - true) ** 2
                mae = abs(predict - true)
                total_rmse.append(rmse)
                total_mae.append(mae)
    RMSE = np.sqrt(np.array(total_rmse).sum())
    MAE = np.array(total_mae).sum()
    print(RMSE, MAE)


if __name__ == '__main__':
    data_path = './Data'
    name_list = ['TaxiBJ', 'BikeNYC', 'TaxiCQ']
    for data_name in name_list:
        print(data_name)
        # ha_test()
        arima_test()