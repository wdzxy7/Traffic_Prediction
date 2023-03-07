import torch
import numpy as np
import torch.nn as nn


class TemporalResCovBlock(nn.Module):
    def __init__(self, dila_rate, padding, causal_cov_size=3, dila_stride=1, in_channel=2, out_channel=2,
                 resnet_layers=5, tcn_kernel_size=3, res_kernel_size=3, data_h=32, data_w=32):
        super(TemporalResCovBlock, self).__init__()
        # parameter
        self.dila_rate = dila_rate
        self.pad = padding
        self.causal_cov_size = causal_cov_size
        self.dila_stride = dila_stride
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.resnet_layers = resnet_layers
        self.tcn_kernel_size = tcn_kernel_size
        self.res_kernel_size = res_kernel_size
        self.data_h = data_h
        self.data_w = data_w
        self.h_pad, self.w_pad = self.cal_padding()
        # module
        # tcn 3 -> 1
        self.dila_conv = nn.Conv3d(self.in_channel, self.out_channel,
                                   kernel_size=(self.causal_cov_size, self.tcn_kernel_size, self.tcn_kernel_size),
                                   padding=(0, self.h_pad, self.w_pad), stride=(1, 1, 1), dilation=(dila_rate, 1, 1),
                                   bias=False)
        self.bn = nn.BatchNorm3d(self.out_channel)
        self.relu = nn.LeakyReLU(inplace=True)
        self.resUnit = ResUnit(out_channel, out_channel, self.data_h, self.data_w, self.res_kernel_size)

    def forward(self, inputs):
        output = self.dila_conv(inputs)
        output = self.bn(output)
        output = self.relu(output)
        for i in range(self.resnet_layers):
            output = self.resUnit(output)
        return output

    def cal_padding(self):
        h_pad = (self.data_h - 1) * 1 + self.tcn_kernel_size - self.data_h
        w_pad = (self.data_w - 1) * 1 + self.tcn_kernel_size - self.data_w
        return int(np.ceil(h_pad / 2)), int(np.ceil(w_pad / 2))


class ResUnit(nn.Module):
    def __init__(self, in_channel, out_channel, data_h, data_w, res_kernel_size):
        super(ResUnit, self).__init__()
        self.data_h = data_h
        self.data_w = data_w
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.res_kernel_size = res_kernel_size
        self.h_pad, self.w_pad = self.cal_padding()
        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size=(1, self.res_kernel_size, self.res_kernel_size),
                              padding=(0, self.h_pad, self.w_pad), stride=(1, 1, 1))
        self.bn = nn.BatchNorm3d(in_channel)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, inputs):
        output = self.conv(inputs)
        output = self.bn(output)
        output = self.relu(output)
        return output + inputs

    def cal_padding(self):
        h_pad = (self.data_h - 1) * 1 + self.res_kernel_size - self.data_h
        w_pad = (self.data_w - 1) * 1 + self.res_kernel_size - self.data_w
        return int(np.ceil(h_pad / 2)), int(np.ceil(w_pad / 2))


class TemporalConvNet(nn.Module):
    def __init__(self, dila_rate_list=None, dila_stride=1, tcn_kernel_size=3, wind_size=7*48, resnet_layers=5,
                 in_channel=None, out_channel=None, res_kernel_size=3, data_h=32, data_w=32):
        super(TemporalConvNet, self).__init__()
        if dila_rate_list is None:
            dila_rate_list = [1, 1, 2, 4, 8, 8, 16, 32, 32, 64]
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.causal_cov_size = 3
        self.causal_layer = len(dila_rate_list)
        self.tcn_kernel_size = tcn_kernel_size
        self.wind_size = wind_size
        self.dila_rate_list = dila_rate_list
        self.padding_rate = self.cal_padding()
        self.dila_stride = dila_stride
        self.resnet_layers = resnet_layers
        self.TemporalRes_blocks = nn.ModuleList()
        self.res_kernel_size = res_kernel_size
        self.data_h = data_h
        self.data_w = data_w
        for dila_rate, padding, c_in, c_out in zip(self.dila_rate_list, self.padding_rate, self.in_channel, self.out_channel):
            self.TemporalRes_blocks.append(TemporalResCovBlock(dila_rate, padding, self.causal_cov_size, self.dila_stride,
                                                               c_in, c_out, self.resnet_layers, self.tcn_kernel_size,
                                                               self.res_kernel_size, self.data_h, self.data_w))

    def forward(self, inputs):
        # padding data seq from 336 to 337 before training
        padded_inputs = self.pad_input(inputs)
        for i in range(len(self.dila_rate_list)):
            output = self.TemporalRes_blocks[i](padded_inputs)
            padded_inputs = output
        return output

    def pad_input(self, inputs):
        shapes = list(inputs.shape)
        shapes[2] = 1
        pad_tensor = torch.zeros(shapes)
        pad_tensor = pad_tensor.to(device=inputs.device)
        padded_inputs = torch.cat((inputs, pad_tensor), 2)
        return padded_inputs

    def cal_padding(self):
        padding_list = []
        for dila in self.dila_rate_list:
            pad = (dila - 1) * (self.causal_cov_size - 1) + self.causal_cov_size - 1
            padding_list.append(pad)
        return padding_list


class SeqAndExcNet(nn.Module):
    def __init__(self, wind_size, batch_size, r):
        super(SeqAndExcNet, self).__init__()
        self.wind_size = wind_size
        self.batch_size = batch_size
        mid_layer = int(np.ceil(self.wind_size / r))
        self.squeeze_layer = nn.AdaptiveAvgPool3d((wind_size, 1, 1))
        self.linear1 = nn.Linear(wind_size, mid_layer)
        self.linear2 = nn.Linear(mid_layer, wind_size)
        self.relu = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        output = self.squeeze_layer(inputs)
        output = output.view(self.batch_size, 2, self.wind_size)
        output1, output2 = output.chunk(2, 1)
        outputs = [output1, output2]
        res_out = []
        for i in range(0, 2):
            c_out = self.linear1(outputs[i])
            c_out = self.relu(c_out)
            c_out = self.linear2(c_out)
            c_out = self.sigmoid(c_out)
            res_out.append(c_out)
        res_out = torch.stack(res_out, dim=1).view(self.batch_size, 2, self.wind_size, 1, 1)
        return torch.mul(inputs, res_out)


class TestModule(nn.Module):
    def __init__(self, wind_size=7*48, batch_size=2, sqe_rate=3, dila_rate_list=None, tcn_kernel_size=3, resnet_layers=5,
                 in_channel=None, out_channel=None, res_kernel_size=3, data_h=32, data_w=32):
        super(TestModule, self).__init__()
        # parameter
        # global
        self.batch_size = batch_size
        self.wind_size = wind_size
        self.sqe_rate = sqe_rate
        # tcn
        if in_channel is None:
            self.in_channel = [2, 4, 8, 16, 32, 64, 32, 16, 8, 4]
        else:
            self.in_channel = in_channel
        if out_channel is None:
            self.out_channel = [4, 8, 16, 32, 64, 32, 16, 8, 4, 2]
        else:
            self.out_channel = out_channel
        self.dila_stride = 1  # static
        self.dila_rate_list = dila_rate_list
        self.tcn_kernel_size = tcn_kernel_size
        self.res_kernel_size = res_kernel_size
        self.data_h = data_h
        self.data_w = data_w
        # resnet
        self.resnet_layers = resnet_layers
        self.SEN_Net = SeqAndExcNet(self.wind_size, self.batch_size, self.sqe_rate)
        self.Tempora_Net = TemporalConvNet(dila_rate_list=self.dila_rate_list, dila_stride=self.dila_stride, tcn_kernel_size=self.tcn_kernel_size,
                                           wind_size=self.wind_size, resnet_layers=self.resnet_layers, in_channel=self.in_channel, out_channel=self.out_channel,
                                           res_kernel_size=self.res_kernel_size, data_h=self.data_h, data_w=self.data_w)

    def forward(self, inputs, ext):
        sened_inputs = self.SEN_Net(inputs)
        y_pre = self.Tempora_Net(sened_inputs)
        return y_pre
