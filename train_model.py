import torch
import numpy as np
import torch.nn as nn
from utils import FlowDataset
import torch.utils.data as data
from test_module import TestModule


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
    device = torch.device(0)
    model = TestModule(7 * 48, batch_size, 3, dila_rate_list=None, kernel_size=3, resnet_layers=resnet_layers)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    show_parameter(model)
    for i in range(epochs):
        stepLR.step()
        for _, batch_data in enumerate(train_loader, 1):
            x_data = batch_data[0].to(device)
            y_data = batch_data[1].to(device)
            ext_data = batch_data[2].to(device)
            y_hat = model(x_data, ext_data)
            loss = criterion(y_hat, y_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('\tTRAINDATE:', '\tEpoch:{}\t\t loss:{}'.format(i, loss.item()))
        test_model(i, model, criterion, val_loader, test_loader, device)


def test_model(i, model, criterion, val_loader, test_loader, device):
    with torch.no_grad():
        model.eval()
        val_RMSE, loss = cal_rmse(model, criterion, val_loader, device)
        print('\tVALIDATE'.ljust(12), '\tEpoch:{}\t\tRMSE:     {} \t loss:{}'.format(i, val_RMSE, loss))
        test_RMSE, loss = cal_rmse(model, criterion, test_loader, device)
        print('\tVALIDATE'.ljust(12), '\tEpoch:{}\t\tRMSE:     {} \t loss:{}'.format(i, test_RMSE, loss))


def cal_rmse(model, criterion, val_loader, device):
    total_loss = []
    model.eval()
    count_all_pred_val = 0
    with torch.no_grad():
        for _, batch_data in enumerate(val_loader):
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
    img = (img + 1.) / 2.
    img = 1. * img * (1292 - 0) + 0.0
    return img


def show_parameter(model):
    par = list(model.parameters())
    s = sum([np.prod(list(d.size())) for d in par])
    print("Parameter of stcnn:", s)


if __name__ == '__main__':
    batch_size = 32
    resnet_layers = 10
    lr = 6e-4
    epochs = 50
    train()