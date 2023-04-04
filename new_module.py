import torch
import numpy as np
import torch.nn as nn


class ResUnit(nn.Module):
    def __init__(self, in_channel, out_channel, data_h, data_w, res_kernel_size):
        super(ResUnit, self).__init__()
        self.data_h = data_h
        self.data_w = data_w
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.res_kernel_size = res_kernel_size
        self.h_pad, self.w_pad = self.cal_padding()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(self.res_kernel_size, self.res_kernel_size),
                               padding=(self.h_pad, self.w_pad), stride=(1, 1))
        self.bn = nn.BatchNorm2d(in_channel)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=(self.res_kernel_size, self.res_kernel_size),
                               padding=(self.h_pad, self.w_pad), stride=(1, 1))
        self.relu = nn.LeakyReLU(inplace=False)
        self.drop = nn.Dropout()

    def forward(self, inputs):
        output = self.relu(inputs)
        output1 = self.conv1(output)
        output = self.bn(output1)
        output = self.conv2(output)
        output = output + output1
        return output + inputs


    def cal_padding(self):
        h_pad = (self.data_h - 1) * 1 + self.res_kernel_size - self.data_h
        w_pad = (self.data_w - 1) * 1 + self.res_kernel_size - self.data_w
        return int(np.ceil(h_pad / 2)), int(np.ceil(w_pad / 2))


class DilaResUnit(nn.Module):
    def __init__(self, in_channel, out_channel, data_h, data_w, res_kernel_size):
        super(DilaResUnit, self).__init__()
        self.data_h = data_h
        self.data_w = data_w
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.res_kernel_size = res_kernel_size
        self.h_pad, self.w_pad = self.cal_padding()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(self.res_kernel_size, self.res_kernel_size),
                               padding=(self.h_pad, self.w_pad), stride=(1, 1), dilation=2)
        self.bn = nn.BatchNorm2d(in_channel)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=(self.res_kernel_size, self.res_kernel_size),
                               padding=(self.h_pad, self.w_pad), stride=(1, 1), dilation=2)
        self.relu = nn.LeakyReLU(inplace=False)
        self.drop = nn.Dropout()

    def forward(self, inputs):
        output = self.relu(inputs)
        output1 = self.conv1(output)
        output = self.bn(output1)
        output = self.conv2(output)
        output = output + output1
        return output + inputs

    def cal_padding(self):
        h_pad = (self.data_h - 1) * 1 + self.res_kernel_size - self.data_h
        w_pad = (self.data_w - 1) * 1 + self.res_kernel_size - self.data_w
        return int(np.ceil(h_pad / 2)) + 1, int(np.ceil(w_pad / 2)) + 1


class CovBlockAttentionNet(nn.Module):
    def __init__(self, wind_size, r, sqe_kernel_size, data_h, data_w):
        super(CovBlockAttentionNet, self).__init__()
        self.wind_size = wind_size
        self.data_h = data_h
        self.data_w = data_w
        self.sqe_kernel_size = sqe_kernel_size
        self.avg_linear_squeeze_layer = nn.AdaptiveAvgPool2d((1, 1))
        self.max_linear_squeeze_layer = nn.AdaptiveMaxPool2d((1, 1))
        self.linear_squeeze_layers = [self.avg_linear_squeeze_layer, self.max_linear_squeeze_layer]
        self.avg_cov_squeeze_layer = nn.AdaptiveAvgPool3d((1, data_h, data_w))
        self.max_cov_squeeze_layer = nn.AdaptiveMaxPool3d((1, data_h, data_w))
        self.mid_layer = int(np.ceil(self.wind_size / r))
        self.avg_linear1 = nn.Linear(wind_size, self.mid_layer)
        self.avg_linear2 = nn.Linear(self.mid_layer, wind_size)
        self.max_linear1 = nn.Linear(wind_size, self.mid_layer)
        self.max_linear2 = nn.Linear(self.mid_layer, wind_size)
        self.am_layers = [[self.avg_linear1, self.avg_linear2], [self.max_linear1, self.max_linear2]]
        self.h_pad, self.w_pad = self.cal_padding()
        self.cov = nn.Conv2d(2, 1, kernel_size=(self.sqe_kernel_size, self.sqe_kernel_size), stride=1,
                             padding=(self.h_pad, self.w_pad))
        self.relu = nn.LeakyReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        res_out = self.linear_squeeze(inputs)
        inputs = torch.mul(inputs, res_out.view(res_out.shape[0], res_out.shape[1], 1, 1))
        res_out = self.cov_squeeze(inputs)
        return torch.mul(inputs, res_out)

    def linear_squeeze(self, inputs):
        linear_squeeze_res = []
        for linear_layer, am_layer in zip(self.linear_squeeze_layers, self.am_layers):
            output = linear_layer(inputs)
            batch_size = inputs.shape[0]
            output = output.view(batch_size, self.wind_size)
            c_out = am_layer[0](output)
            c_out = self.relu(c_out)
            c_out = am_layer[1](c_out)
            c_out = self.sigmoid(c_out)
            linear_squeeze_res.append(c_out)
        res = linear_squeeze_res[0] + linear_squeeze_res[1]
        res = self.sigmoid(res)
        return res

    def cov_squeeze(self, inputs):
        batch_size = inputs.shape[0]
        max_pool = self.max_cov_squeeze_layer(inputs)
        avg_pool = self.avg_cov_squeeze_layer(inputs)
        concat_pool = torch.stack([max_pool, avg_pool], dim=2).view(batch_size, 2, self.data_h, self.data_w)
        output = self.cov(concat_pool)
        output = self.sigmoid(output)
        return output

    def cal_padding(self):
        h_pad = (self.data_h - 1) * 1 + self.sqe_kernel_size - self.data_h
        w_pad = (self.data_w - 1) * 1 + self.sqe_kernel_size - self.data_w
        return int(np.ceil(h_pad / 2)), int(np.ceil(w_pad / 2))


class ExtEmb(nn.Module):
    def __init__(self, data_h, data_w):
        super(ExtEmb, self).__init__()
        self.data_h = data_h
        self.data_w = data_w
        self.emb = nn.Embedding(num_embeddings=4, embedding_dim=1)
        self.linear1 = nn.Linear(4 * 28, 28)
        self.relu = nn.LeakyReLU(inplace=False)
        self.linear2 = nn.Linear(28, 10)
        self.linear3 = nn.Linear(10, 2 * self.data_h * self.data_w)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, ext):
        shape = ext.shape
        embed_ext = self.emb(ext)
        linear_ext = embed_ext.view(shape[0], -1)
        output = self.linear1(linear_ext)
        output = self.drop(output)
        output = self.relu(output)
        output = self.linear2(output)
        output = self.relu(output)
        output = self.linear3(output)
        output = output.view(shape[0], 2, 1, self.data_h, self.data_w)
        return output


class NewModule(nn.Module):
    def __init__(self, sqe_rate=3, sqe_kernel_size=3, resnet_layers=5, res_kernel_size=3, data_h=32, data_w=32, use_ext=True):
        super(NewModule, self).__init__()
        # parameter
        # global
        self.heads = 1
        self.use_ext = use_ext
        # CovBlockAttentionNet
        self.sqe_rate = sqe_rate
        self.sqe_kernel_size = sqe_kernel_size
        # Tcn
        self.data_h = data_h
        self.data_w = data_w
        # Resnet
        self.resnet_layers = resnet_layers
        self.res_kernel_size = res_kernel_size
        self.Input_SEN_Net = CovBlockAttentionNet(64, self.sqe_rate, self.sqe_kernel_size, self.data_h, self.data_w)
        self.tanh = nn.Tanh()
        self.emb = ExtEmb(self.data_h, self.data_w)
        self.up_channel = nn.Sequential(nn.Conv2d(38, 64, 1, 1),
                                        nn.BatchNorm2d(64),
                                        nn.LeakyReLU(inplace=True))
        self.Res_Net = self.build_resnet()
        self.Dila_Net = self.build_dila_resnet()
        self.Sen_Net = CovBlockAttentionNet(64, self.sqe_rate, self.sqe_kernel_size, self.data_h, self.data_w)
        self.Out_Net = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=(1, 1)),
                                     nn.BatchNorm2d(32),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=(1, 1)),
                                     nn.BatchNorm2d(2),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Conv2d(2, 2, kernel_size=1, stride=1),
                                     nn.Tanh(),
                                     nn.Conv2d(2, 2, kernel_size=1, stride=1),
                                     nn.Tanh()
                                     )

    def forward(self, inputs, ext):
        if self.use_ext:
            ext = self.emb(ext)
        inputs = self.merge_data(inputs, ext)
        output = self.up_channel(inputs)
        res_out = output
        dila_out = output
        for i in range(self.resnet_layers):
            res_out = self.Res_Net[i](res_out)
            dila_out = self.Dila_Net[i](dila_out)
        output = res_out + dila_out
        output = self.Out_Net(output)
        return output.view(inputs.shape[0], 2, 1, self.data_h, self.data_w)

    def merge_data(self, data, ext):
        current_data = []
        leak_data = []
        day_data = []
        T = 48
        for i in range(1, 8):
            leak_data.append(data[:, :, 336 - i * T:337 - i * T, :, :])
            day_data.append(data[:, :, 335 - (i - 1) * T:336 - (i - 1) * T, :, :])
            if i < 5:
                current_data.append(data[:, :, 336 - i:337 - i, :, :])
        leak_data.append(ext)
        data = torch.stack(current_data + day_data + leak_data, dim=1)
        shape = data.shape
        return data.view(shape[0], shape[1] * shape[2], shape[4], shape[5])

    def build_resnet(self):
        res_net = nn.ModuleList()
        for i in range(self.resnet_layers):
            res_net.append(ResUnit(64, 64, self.data_h, self.data_w, self.res_kernel_size))
        return res_net

    def build_sennet(self):
        sen_net = nn.ModuleList()
        for i in range(self.resnet_layers):
            sen_net.append(CovBlockAttentionNet(64, self.sqe_rate, self.sqe_kernel_size, self.data_h, self.data_w))
        return sen_net

    def build_dila_resnet(self):
        dila_net = nn.ModuleList()
        for i in range(self.resnet_layers):
            dila_net.append(DilaResUnit(64, 64, self.data_h, self.data_w, self.res_kernel_size))
        return dila_net