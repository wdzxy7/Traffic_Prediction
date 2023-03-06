import torch
import numpy as np
import torch.nn as nn


class TemporalResCovBlock(nn.Module):
    def __init__(self, dila_rate, padding, causal_cov_size=3, dila_stride=1, in_channel=2, out_channel=2,
                 resnet_layers=5):
        super(TemporalResCovBlock, self).__init__()
        # parameter
        self.dila_rate = dila_rate
        self.pad = padding
        self.causal_cov_size = causal_cov_size
        self.dila_stride = dila_stride
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.resnet_layers = resnet_layers
        # module
        # 3 -> 1
        self.dila_conv = nn.Conv3d(self.in_channel, self.out_channel, kernel_size=(3, 3, 3), padding=(0, 1, 1),
                                   stride=(1, 1, 1), dilation=(dila_rate, 1, 1), bias=False)
        self.bn = nn.BatchNorm3d(self.out_channel)
        self.relu = nn.LeakyReLU(inplace=True)
        self.resUnit = ResUnit(out_channel, out_channel)

    def forward(self, inputs):
        output = self.dila_conv(inputs)
        output = self.bn(output)
        output = self.relu(output)
        print(output.shape)
        for i in range(self.resnet_layers):
            output = self.resUnit(output)
        return output


class ResUnit(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResUnit, self).__init__()
        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 1, 1))
        self.bn = nn.BatchNorm3d(in_channel)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, inputs):
        output = self.conv(inputs)
        output = self.bn(output)
        output = self.relu(output)
        return output + inputs


class TemporalConvNet(nn.Module):
    def __init__(self, dila_rate_list=None, dila_stride=1, kernel_size=3, wind_size=7*48, resnet_layers=5,
                 in_channel=None, out_channel=None):
        super(TemporalConvNet, self).__init__()
        if dila_rate_list is None:
            dila_rate_list = [1, 1, 2, 4, 8, 8, 16, 32, 32, 64]
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.causal_cov_size = 3
        self.causal_layer = len(dila_rate_list)
        self.kernel_size = kernel_size
        self.wind_size = wind_size
        self.dila_rate_list = dila_rate_list
        self.padding_rate = self.cal_padding()
        self.dila_stride = dila_stride
        self.resnet_layers = resnet_layers
        self.TemporalRes_blocks = nn.ModuleList()
        for dila_rate, padding, c_in, c_out in zip(self.dila_rate_list, self.padding_rate, self.in_channel, self.out_channel):
            self.TemporalRes_blocks.append(TemporalResCovBlock(dila_rate, padding, self.causal_cov_size, self.dila_stride,
                                                               c_in, c_out, self.resnet_layers))

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
    def __init__(self, wind_size=7*48, batch_size=2, sqe_rate=3, dila_rate_list=None, kernel_size=3, resnet_layers=5,
                 in_channel=None, out_channel=None):
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
        self.kernel_size = kernel_size
        # resnet
        self.resnet_layers = resnet_layers
        self.SEN_Net = SeqAndExcNet(self.wind_size, self.batch_size, self.sqe_rate)
        self.Tempora_Net = TemporalConvNet(None, self.dila_rate_list, self.dila_stride, self.kernel_size, self.resnet_layers,
                                           self.in_channel, self.out_channel)

    def forward(self, inputs, ext):
        sened_inputs = self.SEN_Net(inputs)
        y_pre = self.Tempora_Net(sened_inputs)
        return y_pre
