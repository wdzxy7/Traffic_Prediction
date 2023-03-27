import os
import random
import sys
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
from utils import FlowDataset
import torch.utils.data as data
from new_module import NewModule


'''
use temp code rmse 17.26
seed 6540
lr 7e-4
sqe_rate 2
'''

parser = argparse.ArgumentParser(description='Parameters for my module')
parser.add_argument('--epochs', type=int, default=300, help='Epochs of train')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size of dataloader')
parser.add_argument('--lr', type=float, default=7e-4, help='Learning rate of optimizer')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight_decay of optimizer')
parser.add_argument('--sqe_rate', type=int, default=2, help='The squeeze rate of CovBlockAttentionNet')
parser.add_argument('--sqe_kernel_size', type=int, default=3, help='The kernel size of CovBlockAttentionNet')
parser.add_argument('--week_resnet_layers', type=int, default=2, help='Number of layers of week and day data in ResNet') # 2
parser.add_argument('--current_resnet_layers', type=int, default=1, help='Number of layers of current data in ResNet')
parser.add_argument('--tcn_kernel_size', type=int, default=3, help='TCN convolution kernel size')
parser.add_argument('--res_kernel_size', type=int, default=3, help='ResUnit kernel size')
parser.add_argument('--load', type=bool, default=False, help='Whether load checkpoint')
parser.add_argument('--check_point', type=int, default=False, help='Checkpoint')
parser.add_argument('--data_name', type=str, default='TaxiBJ', help='Train data name')
parser.add_argument('--use_ext', type=bool, default=True, help='Whether use external data')


def load_data():
    train_dataset = FlowDataset(data_name, data_type='train', use_ext=use_ext)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    val_dataset = FlowDataset(data_name, data_type='val', use_ext=use_ext)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = FlowDataset(data_name, data_type='test', use_ext=use_ext)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def test():
    model_path = "./model/model_rmse_TaxiBJ_parameter_test2.pkl"
    train_loader, val_loader, test_loader = load_data()
    model = NewModule()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    criterion = nn.MSELoss()
    show_parameter(model)
    test_model(model, criterion, val_loader, test_loader)


def test_model(model, criterion, val_loader, test_loader):
    with torch.no_grad():
        model.eval()
        val_RMSE, val_MAE, loss = cal_rmse(model, criterion, val_loader)
        print('\n')
        print('\tVALIDATE'.ljust(12), '\tEpoch:{}\t\tRMSE:     {} \t\tMAE:     {} \t loss:{}'.format(0, val_RMSE, val_MAE, loss))
        test_RMSE, test_MAE, loss = cal_rmse(model, criterion, test_loader)
        print('\tTESTDATE'.ljust(12), '\tEpoch:{}\t\tRMSE:     {} \t\tMAE:     {} \t loss:{}'.format(0, test_RMSE, test_MAE, loss))


def cal_rmse(model, criterion, data_loader):
    rmse_total_loss = []
    mae_total_loss = []
    model.eval()
    count_all_pred_val = 0
    with torch.no_grad():
        for _, batch_data in enumerate(data_loader):
            x_data = batch_data[0].to(device)
            y_data = batch_data[1].to(device)
            ext_data = batch_data[2].to(device)
            y_hat = model(x_data, ext_data)
            criterion_loss = criterion(y_hat, y_data)
            y_real = inverse_mmn(y_data).float()
            y_hat_real = inverse_mmn(y_hat).float()
            rmse_loss = (y_hat_real - y_real) ** 2
            mae_loss = abs(y_hat_real - y_real)
            rmse_total_loss.append(rmse_loss.sum().item())
            mae_total_loss.append(mae_loss.sum().item())
            count_all_pred_val += y_hat_real.shape.numel()
    RMSE = np.sqrt(np.array(rmse_total_loss).sum() / count_all_pred_val)
    MAE = np.array(mae_total_loss).sum() / count_all_pred_val
    return RMSE, MAE, criterion_loss.item()


def inverse_mmn(img):
    data_max = 1292
    data_min = 0
    if data_name == 'Taxi_Bj':
        data_max = 1292
        data_min = 0
    elif data_name == 'Bike_NYC':
        data_max = 267
        data_min = 0
    elif data_name == 'Taxi_NYC':
        data_max = 1852
        data_min = 0
    img = (img + 1.) / 2.
    img = 1. * img * (data_max - data_min) + data_min
    return img


def show_parameter(model):
    par = list(model.parameters())
    s = sum([np.prod(list(d.size())) for d in par])
    print("Parameter of 3DRTCN:", s)


if __name__ == '__main__':
    device = torch.device(2)
    args = parser.parse_args()
    sqe_rate = args.sqe_rate
    sqe_kernel_size = args.sqe_kernel_size
    batch_size = args.batch_size
    week_resnet_layers = args.week_resnet_layers
    current_resnet_layers = args.current_resnet_layers
    tcn_kernel_size = args.tcn_kernel_size
    res_kernel_size = args.res_kernel_size
    lr = args.lr
    weight_decay = args.weight_decay
    epochs = args.epochs
    load = args.load
    check_point = args.check_point
    data_name = args.data_name
    use_ext = args.use_ext
    logger = logging.getLogger(__name__)
    test()