import os
import sys
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
from utils import FlowDataset
import torch.utils.data as data
from test_module import TestModule

parser = argparse.ArgumentParser(description='parameters for my module')
parser.add_argument('--epochs', type=int, default=50, help='Epochs of train')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size of dataloader')
parser.add_argument('--lr', type=float, default=6e-4, help='Learning rate of optimizer')
parser.add_argument('--sqe_rate', type=int, default=4, help='The squeeze rate of CovBlockAttentionNet')
parser.add_argument('--sqe_kernel_size', type=int, default=7, help='The kernel size of CovBlockAttentionNet')
parser.add_argument('--week_resnet_layers', type=int, default=5, help='Number of layers of week and day data in ResNet')
parser.add_argument('--current_resnet_layers', type=int, default=10, help='Number of layers of current data in ResNet')
parser.add_argument('--tcn_kernel_size', type=int, default=3, help='TCN convolution kernel size')
parser.add_argument('--res_kernel_size', type=int, default=3, help='ResUnit kernel size')
parser.add_argument('--data_h', type=int, default=32, help='The high of one data')
parser.add_argument('--data_w', type=int, default=32, help='The width of one data')


def load_data():
    train_dataset = FlowDataset('TaxiBJ', data_type='train')
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    val_dataset = FlowDataset('TaxiBJ', data_type='val')
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = FlowDataset('TaxiBJ', data_type='test')
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train():
    train_loader, val_loader, test_loader = load_data()
    model = TestModule(wind_size=7 * 48, batch_size=batch_size, sqe_rate=sqe_rate, sqe_kernel_size=sqe_kernel_size, dila_rate_list=None,
                       tcn_kernel_size=tcn_kernel_size,  week_resnet_layers=week_resnet_layers, res_kernel_size=res_kernel_size,
                       current_resnet_layer=current_resnet_layers, data_h=data_h, data_w=data_w)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    show_parameter(model)
    train_len = len(train_loader)
    print(len(train_loader), len(test_loader), len(val_loader))
    for i in range(epochs):
        for _, batch_data in enumerate(train_loader, 1):
            x_data = batch_data[0].to(device)
            y_data = batch_data[1].to(device)
            ext_data = batch_data[2].to(device)
            y_hat = model(x_data, ext_data)
            loss = criterion(y_hat, y_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sys.stdout.write("\rTRAINDATE:  Epoch:{}\t\t loss:{} res train:{}".format(i, loss.item(), train_len - _))
        test_model(i, model, criterion, val_loader, test_loader)
        stepLR.step()


def test_model(i, model, criterion, val_loader, test_loader):
    global min_rmse
    with torch.no_grad():
        model.eval()
        val_RMSE, loss = cal_rmse(model, criterion, val_loader)
        print('\tVALIDATE'.ljust(12), '\tEpoch:{}\t\tRMSE:     {} \t loss:{}'.format(i, val_RMSE, loss))
        mess = '\tVALIDATE'.ljust(12), '\tEpoch:{}\t\tRMSE:     {} \t loss:{}'.format(i, val_RMSE, loss)
        if val_RMSE < min_rmse:
            min_rmse = val_RMSE
            torch.save(model.state_dict(), "./model_parameter.pkl")
        logger.info(str(mess))
        test_RMSE, loss = cal_rmse(model, criterion, test_loader)
        print('\tTESTDATE'.ljust(12), '\tEpoch:{}\t\tRMSE:     {} \t loss:{}'.format(i, test_RMSE, loss))
        mess = '\tTESTDATE'.ljust(12), '\tEpoch:{}\t\tRMSE:     {} \t loss:{}'.format(i, test_RMSE, loss)
        logger.info(str(mess))


def cal_rmse(model, criterion, data_loader):
    total_loss = []
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
            loss = (y_hat_real - y_real) ** 2
            total_loss.append(loss.sum().item())
            count_all_pred_val += y_hat_real.shape.numel()
            break
    RMSE = np.sqrt(np.array(total_loss).sum() / count_all_pred_val)
    return RMSE, criterion_loss.item()


def inverse_mmn(img):
    img = (img + 1.) / 2.
    img = 1. * img * (1292 - 0) + 0.0
    return img


def show_parameter(model):
    for k, v in model.state_dict().items():
        print(k, v.shape)
    par = list(model.parameters())
    s = sum([np.prod(list(d.size())) for d in par])
    print("Parameter of 3DRTCN:", s)


def set_logger():
    global logger
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("./run_log.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


if __name__ == '__main__':
    min_rmse = 999999
    logger = logging.getLogger(__name__)
    set_logger()
    args = parser.parse_args()
    device = torch.device(1)
    sqe_rate = args.sqe_rate
    sqe_kernel_size = args.sqe_kernel_size
    batch_size = args.batch_size
    week_resnet_layers = args.week_resnet_layers
    current_resnet_layers = args.current_resnet_layers
    tcn_kernel_size = args.tcn_kernel_size
    res_kernel_size = args.res_kernel_size
    data_h = args.data_h
    data_w = args.data_w
    lr = args.lr
    epochs = args.epochs
    train()