import os
import sys
import time
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
from utils import FlowDataset
import torch.utils.data as data
from ResTCN_model import ResTCN


'''
use new module
seed 7687
lr 7e-4
'''

parser = argparse.ArgumentParser(description='Parameters for my module')
parser.add_argument('--epochs', type=int, default=300, help='Epochs of train')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size of dataloader')
parser.add_argument('--lr', type=float, default=9e-4, help='Learning rate of optimizer')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight_decay of optimizer')
parser.add_argument('--sqe_rate', type=int, default=2, help='The squeeze rate of CovBlockAttentionNet')
parser.add_argument('--sqe_kernel_size', type=int, default=3, help='The kernel size of CovBlockAttentionNet')
parser.add_argument('--week_resnet_layers', type=int, default=2, help='Number of layers of week and day data in ResNet') # 2
parser.add_argument('--current_resnet_layers', type=int, default=1, help='Number of layers of current data in ResNet')
parser.add_argument('--tcn_kernel_size', type=int, default=3, help='TCN convolution kernel size')
parser.add_argument('--res_kernel_size', type=int, default=3, help='ResUnit kernel size')
parser.add_argument('--load', type=bool, default=False, help='Whether load checkpoint')
parser.add_argument('--check_point', type=int, default=69, help='Checkpoint')
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


def get_h_w():
    data_h = 0
    data_w = 0
    if data_name == 'TaxiBJ':
        data_h = 32
        data_w = 32
    elif data_name == 'BikeNYC':
        data_h = 16
        data_w = 8
    elif data_name == 'TaxiNYC':
        data_h = 15
        data_w = 5
    return data_h, data_w


def train(load_sign):
    train_loader, val_loader, test_loader = load_data()
    data_h, data_w = get_h_w()
    model = ResTCN(wind_size=7 * 48, batch_size=batch_size, sqe_rate=sqe_rate, sqe_kernel_size=sqe_kernel_size, dila_rate_list=None,
                       tcn_kernel_size=tcn_kernel_size,  week_resnet_layers=week_resnet_layers, res_kernel_size=res_kernel_size,
                       current_resnet_layer=current_resnet_layers, data_h=data_h, data_w=data_w, use_ext=use_ext)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    stepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    show_parameter(model)
    if load_sign:
        model, optimizer = load_checkpoint(model, optimizer)
    train_len = len(train_loader)
    test_count = 1
    for i in range(epochs):
        for _, batch_data in enumerate(train_loader, 1):
            # model.train()
            optimizer.zero_grad()
            x_data = batch_data[0].to(device)
            y_data = batch_data[1].to(device)
            ext_data = batch_data[2].to(device)
            y_hat = model(x_data, ext_data)
            loss = criterion(y_hat, y_data)
            loss.backward()
            optimizer.step()
            sys.stdout.write("\rTRAINDATE:  Epoch:{}\t\t loss:{} res train:{}".format(i, loss.item(), train_len - _))
        test_model(i, model, criterion, val_loader, test_loader)
        if test_count % 5 == 0:
            save_checkpoint(model, i, optimizer)
        test_count += 1
        stepLR.step()


def test_model(i, model, criterion, val_loader, test_loader):
    global min_rmse
    model_path = './model'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    with torch.no_grad():
        model.eval()
        val_RMSE, val_MAE, loss = cal_rmse(model, criterion, val_loader)
        print('\n')
        print('\tVALIDATE'.ljust(12), '\tEpoch:{}\t\tRMSE:     {} \t\tMAE:     {} \t loss:{}'.format(i, val_RMSE, val_MAE, loss))
        mess = '\tVALIDATE'.ljust(12), '\tEpoch:{}\t\tRMSE:     {} \t\tMAE:     {} \t loss:{}'.format(i, val_RMSE, val_MAE, loss)
        if val_RMSE < min_rmse:
            min_rmse = val_RMSE
            path = os.path.join(model_path, sav_dict[key][1].format(data_name))
            torch.save(model.state_dict(), path)
        if val_MAE < min_mae:
            min_rmse = val_RMSE
            path = os.path.join(model_path, sav_dict[key][4].format(data_name))
            torch.save(model.state_dict(), path)
        logger.info(str(mess))
        test_RMSE, test_MAE, loss = cal_rmse(model, criterion, test_loader)
        print('\tTESTDATE'.ljust(12), '\tEpoch:{}\t\tRMSE:     {} \t\tMAE:     {} \t loss:{}'.format(i, test_RMSE, test_MAE, loss))
        mess = '\tTESTDATE'.ljust(12), '\tEpoch:{}\t\tRMSE:     {} \t\tMAE:     {} \t loss:{}'.format(i, test_RMSE, test_MAE, loss)
        logger.info(str(mess))


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


def set_logger():
    global logger
    log_path = './run_log'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(os.path.join(log_path, sav_dict[key][0].format(data_name)))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def save_checkpoint(model, epoch, optimizer):
    check_save_path = './checkpoints/'
    check_path = os.path.join(check_save_path, data_name)
    if not os.path.exists(check_save_path):
        os.makedirs(check_save_path)
    if not os.path.exists(check_path):
        os.makedirs(check_path)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    check_model_path = os.path.join(check_path, sav_dict[key][2].format(data_name, epoch))
    torch.save(checkpoint, check_model_path)


def load_checkpoint(model, optimizer):
    global epochs
    check_path = './checkpoints/' + data_name
    check_model_path = os.path.join(check_path, sav_dict[key][2].format(data_name, check_point))
    train_state = torch.load(check_model_path)
    model.load_state_dict(train_state['model_state_dict'])
    optimizer.load_state_dict(train_state['optimizer_state_dict'])
    epochs = epochs - train_state['epoch'] - 1
    return model, optimizer


if __name__ == '__main__':
    sav_dict = {'root': ['run_{}_log.log', 'model_{}_parameter.pkl', 'model_{}_{:03d}.pt'],
                'test': ['run_{}_log_test.log', 'model_{}_parameter_test.pkl', 'model_{}_{:03d}_test.pt'],
                'test2': ['run_{}_log_test2.log', 'model_{}_parameter_test2.pkl', 'model_{}_{:03d}_test2.pt'],
                'test3': ['run_{}_log_test3.log', 'model_{}_parameter_test3.pkl', 'model_{}_{:03d}_test3.pt'],
                'test4': ['run_{}_log_test4.log', 'model_{}_parameter_test4.pkl', 'model_{}_{:03d}_test4.pt'],
                'test5': ['run_{}_log_test5.log', 'model_{}_parameter_test5.pkl', 'model_{}_{:03d}_test5.pt'],
                'test6': ['run_{}_log_test6.log', 'model_{}_parameter_test6.pkl', 'model_{}_{:03d}_test6.pt']}
    key = 'test4'
    device = torch.device(4)
    min_rmse = 16.25069473853743
    min_mae = 9.865
    print('{}:  lr 9 seed is {}'.format(key, 7687))
    torch.manual_seed(7687)
    torch.cuda.manual_seed(7687)
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
    set_logger()
    train(load)