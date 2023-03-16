import os
import sys
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
from utils import FlowDataset
import torch.utils.data as data
import torch.distributed as dist
from test_module import TestModule
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


parser = argparse.ArgumentParser(description='Parameters for my module')
parser.add_argument('--epochs', type=int, default=50, help='Epochs of train')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size of dataloader')
parser.add_argument('--lr', type=float, default=6e-4, help='Learning rate of optimizer')
parser.add_argument('--sqe_rate', type=int, default=4, help='The squeeze rate of CovBlockAttentionNet')
parser.add_argument('--sqe_kernel_size', type=int, default=3, help='The kernel size of CovBlockAttentionNet')
parser.add_argument('--week_resnet_layers', type=int, default=2, help='Number of layers of week and day data in ResNet')
parser.add_argument('--current_resnet_layers', type=int, default=2, help='Number of layers of current data in ResNet')
parser.add_argument('--tcn_kernel_size', type=int, default=3, help='TCN convolution kernel size')
parser.add_argument('--res_kernel_size', type=int, default=3, help='ResUnit kernel size')
parser.add_argument('--load', type=bool, default=False, help='Whether load checkpoint')
parser.add_argument('--check_point', type=int, default=False, help='Checkpoint')
parser.add_argument('--data_name', type=str, default='TaxiBJ', help='Train data name')
parser.add_argument('--use_ext', type=bool, default=True, help='Whether use external data')
parser.add_argument('--device_ids', type=str, default='0')
parser.add_argument('--local_rank', type=int, default=1)


def load_data():
    train_dataset = FlowDataset(data_name, data_type='train', use_ext=use_ext)
    sampler_train = DistributedSampler(train_dataset)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler_train, num_workers=16)
    val_dataset = FlowDataset(data_name, data_type='val', use_ext=use_ext)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    test_dataset = FlowDataset(data_name, data_type='test', use_ext=use_ext)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    return train_loader, val_loader, test_loader, sampler_train


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
    train_loader, val_loader, test_loader, sampler_train = load_data()
    data_h, data_w = get_h_w()
    model = TestModule(wind_size=7 * 48, batch_size=batch_size, sqe_rate=sqe_rate, sqe_kernel_size=sqe_kernel_size, dila_rate_list=None,
                       tcn_kernel_size=tcn_kernel_size,  week_resnet_layers=week_resnet_layers, res_kernel_size=res_kernel_size,
                       current_resnet_layer=current_resnet_layers, data_h=data_h, data_w=data_w, use_ext=use_ext)
    torch.cuda.set_device(device)
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[device_ids[args.local_rank]], output_device=device_ids[args.local_rank], find_unused_parameters=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    show_parameter(model)
    if load_sign:
        model, optimizer = load_checkpoint(model, optimizer)
    train_len = len(train_loader)
    test_count = 1
    for i in range(epochs):
        sampler_train.set_epoch(i)
        for _, batch_data in enumerate(train_loader, 1):
            x_data = batch_data[0].to(device)
            y_data = batch_data[1].to(device)
            ext_data = batch_data[2].to(device)
            y_hat = model(x_data, ext_data)
            loss = criterion(y_hat, y_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.local_rank == 0:
                sys.stdout.write("\rTRAINDATE:  Epoch:{}\t\t loss:{} res train:{}".format(i, loss.item(), train_len - _))
        if args.local_rank == 0:
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
        val_RMSE, loss = cal_rmse(model, criterion, val_loader)
        print('\n')
        print('\tVALIDATE'.ljust(12), '\tEpoch:{}\t\tRMSE:     {} \t loss:{}'.format(i, val_RMSE, loss))
        mess = '\tVALIDATE'.ljust(12), '\tEpoch:{}\t\tRMSE:     {} \t loss:{}'.format(i, val_RMSE, loss)
        if val_RMSE < min_rmse:
            min_rmse = val_RMSE
            path = os.path.join(model_path, "model_{}_parameter_change.pkl".format(data_name))
            torch.save(model.module.state_dict(), path)
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
    RMSE = np.sqrt(np.array(total_loss).sum() / count_all_pred_val)
    return RMSE, criterion_loss.item()


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
    handler = logging.FileHandler(os.path.join(log_path, "run_{}_log_change.log".format(data_name)))
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
    check_model_path = os.path.join(check_path, "model_{}_{:03d}_change.pt".format(data_name, epoch))
    torch.save(checkpoint, check_model_path)


def load_checkpoint(model, optimizer):
    global epochs
    check_path = './model/'
    check_model_path = os.path.join(check_path, 'model_{:03d}.pt'.format(check_point))
    train_state = torch.load(check_model_path)
    model.load_state_dict(train_state['model_state_dict'])
    optimizer.load_state_dict(train_state['optimizer_state_dict'])
    epochs = epochs - train_state['epoch'] - 1
    return model, optimizer


if __name__ == '__main__':
    min_rmse = 999999
    args = parser.parse_args()
    # device = torch.device(0)
    sqe_rate = args.sqe_rate
    sqe_kernel_size = args.sqe_kernel_size
    batch_size = args.batch_size
    week_resnet_layers = args.week_resnet_layers
    current_resnet_layers = args.current_resnet_layers
    tcn_kernel_size = args.tcn_kernel_size
    res_kernel_size = args.res_kernel_size
    lr = args.lr
    epochs = args.epochs
    load = args.load
    check_point = args.check_point
    data_name = args.data_name
    use_ext = args.use_ext
    logger = logging.getLogger(__name__)
    set_logger()
    device_ids = list(map(int, args.device_ids.split(',')))
    dist.init_process_group(backend='nccl')
    device = torch.device('cuda:{}'.format(device_ids[args.local_rank]))
    train(load)