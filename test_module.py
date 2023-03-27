import torch
import numpy as np
import torch.nn as nn


class TemporalResCovBlock(nn.Module):
    def __init__(self, causal_cov_size=3, dila_rate=4, dila_stride=1, in_channel=2, out_channel=2,
                 resnet_layers=5, tcn_kernel_size=3, res_kernel_size=3, data_h=32, data_w=32):
        super(TemporalResCovBlock, self).__init__()
        # parameter
        self.causal_cov_size = causal_cov_size
        self.dila_rate = dila_rate
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
                                   padding=(0, self.h_pad, self.w_pad), stride=(1, 1, 1), dilation=(self.dila_rate, 1, 1),
                                   bias=False)
        self.bn = nn.BatchNorm3d(self.out_channel)
        self.relu = nn.LeakyReLU(inplace=False)
        self.resnet = nn.ModuleList()
        self.build_resnet()

    def forward(self, inputs):
        output = self.dila_conv(inputs)
        output = self.bn(output)
        output = self.relu(output)
        for unit in self.resnet:
            output = unit(output)
        return output

    def cal_padding(self):
        h_pad = (self.data_h - 1) * 1 + self.tcn_kernel_size - self.data_h
        w_pad = (self.data_w - 1) * 1 + self.tcn_kernel_size - self.data_w
        return int(np.ceil(h_pad / 2)), int(np.ceil(w_pad / 2))

    def build_resnet(self):
        for i in range(self.resnet_layers):
            res_unit = ResUnit(self.out_channel, self.out_channel, self.data_h, self.data_w, self.res_kernel_size)
            self.resnet.append(res_unit)


class ResUnit(nn.Module):
    def __init__(self, in_channel, out_channel, data_h, data_w, res_kernel_size):
        super(ResUnit, self).__init__()
        self.data_h = data_h
        self.data_w = data_w
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.res_kernel_size = res_kernel_size
        self.h_pad, self.w_pad = self.cal_padding()
        # in channel == out channel
        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size=(1, self.res_kernel_size, self.res_kernel_size),
                              padding=(0, self.h_pad, self.w_pad), stride=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(in_channel)
        self.conv2 = nn.Conv3d(in_channel, out_channel, kernel_size=(1, self.res_kernel_size, self.res_kernel_size),
                              padding=(0, self.h_pad, self.w_pad), stride=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(in_channel)
        self.relu = nn.LeakyReLU(inplace=False)
        self.drop = nn.Dropout()

    def forward(self, inputs):
        output = self.relu(inputs)
        output = self.bn1(output)
        output = self.conv1(output)
        output = self.drop(output)
        output = self.relu(output)
        output = self.bn2(output)
        output = self.conv2(output)
        output = self.drop(output)
        return output + inputs

    def cal_padding(self):
        h_pad = (self.data_h - 1) * 1 + self.res_kernel_size - self.data_h
        w_pad = (self.data_w - 1) * 1 + self.res_kernel_size - self.data_w
        return int(np.ceil(h_pad / 2)), int(np.ceil(w_pad / 2))


class TemporalConvNet(nn.Module):
    def __init__(self, dila_rate_type='week', dila_stride=1, tcn_kernel_size=3, wind_size=7*48, resnet_layers=5,
                 in_channel=None, out_channel=None, res_kernel_size=3, data_h=32, data_w=32):
        super(TemporalConvNet, self).__init__()
        self.dila_rate_type = dila_rate_type
        if dila_rate_type == 'week':
            self.causal_cov_size_list = [32, 32, 16, 8, 4, 4, 3, 3]
            self.dila_rate = [4, 4, 3, 3, 2, 3, 2, 1]
        else:
            self.causal_cov_size_list = [8, 4, 4, 4, 3]
            self.dila_rate = [3, 3, 3, 2, 1]
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.causal_layer = len(self.causal_cov_size_list)
        self.tcn_kernel_size = tcn_kernel_size
        self.wind_size = wind_size
        self.causal_cov_size = self.causal_cov_size_list
        # self.padding_rate = self.cal_padding()
        self.dila_stride = dila_stride
        self.resnet_layers = resnet_layers
        self.TemporalRes_blocks = nn.ModuleList()
        self.res_kernel_size = res_kernel_size
        self.data_h = data_h
        self.data_w = data_w
        for causal_cov, dila_rate, c_in, c_out in zip(self.causal_cov_size_list, self.dila_rate, self.in_channel, self.out_channel):
            self.TemporalRes_blocks.append(TemporalResCovBlock(causal_cov, dila_rate, self.dila_stride,
                                                               c_in, c_out, self.resnet_layers, self.tcn_kernel_size,
                                                               self.res_kernel_size, self.data_h, self.data_w))

    def forward(self, inputs):
        # padding data seq from 336 to 337 before training
        # padded_inputs = self.pad_input(inputs)
        output = inputs
        for i in range(len(self.causal_cov_size_list)):
            output = self.TemporalRes_blocks[i](output)
        return output

    def pad_input(self, inputs):
        if self.dila_rate_type == 'week':
            shapes = list(inputs.shape)
            shapes[2] = 1
        else:
            shapes = list(inputs.shape)
            shapes[2] = 5
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


class CovBlockAttentionNet(nn.Module):
    def __init__(self, wind_size, r, sqe_kernel_size, data_h, data_w):
        super(CovBlockAttentionNet, self).__init__()
        self.wind_size = wind_size
        self.data_h = data_h
        self.data_w = data_w
        self.sqe_kernel_size = sqe_kernel_size
        self.avg_linear_squeeze_layer = nn.AdaptiveAvgPool3d((wind_size, 1, 1))
        self.max_linear_squeeze_layer = nn.AdaptiveMaxPool3d((wind_size, 1, 1))
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
        self.cov = nn.Conv3d(2, 1, kernel_size=(2, self.sqe_kernel_size, self.sqe_kernel_size), stride=(1, 1, 1),
                             padding=(0, self.h_pad, self.w_pad))
        self.relu = nn.LeakyReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        res_out = self.linear_squeeze(inputs)
        inputs = torch.mul(inputs, res_out)
        res_out = self.cov_squeeze(inputs)
        return torch.mul(inputs, res_out)

    def linear_squeeze(self, inputs):
        linear_squeeze_res = []
        for linear_layer, am_layer in zip(self.linear_squeeze_layers, self.am_layers):
            output = linear_layer(inputs)
            batch_size = inputs.shape[0]
            output = output.view(batch_size, 2, self.wind_size)
            output1, output2 = output.chunk(2, 1)
            outputs = [output1, output2]
            res_out = []
            for i in range(0, 2):
                c_out = am_layer[0](outputs[i])
                c_out = self.relu(c_out)
                c_out = am_layer[1](c_out)
                c_out = self.sigmoid(c_out)
                res_out.append(c_out)
            res_out = torch.stack(res_out, dim=1).view(batch_size, 2, self.wind_size, 1, 1)
            linear_squeeze_res.append(res_out)
        res = linear_squeeze_res[0] + linear_squeeze_res[1]
        res = self.sigmoid(res)
        return res

    def cov_squeeze(self, inputs):
        batch_size = inputs.shape[0]
        max_pool = self.max_cov_squeeze_layer(inputs)
        avg_pool = self.avg_cov_squeeze_layer(inputs)
        concat_pool = torch.stack([max_pool, avg_pool], dim=2).view(batch_size, 2, 2, self.data_h, self.data_w)
        output = self.cov(concat_pool)
        output = self.sigmoid(output)
        return output

    def cal_padding(self):
        h_pad = (self.data_h - 1) * 1 + self.sqe_kernel_size - self.data_h
        w_pad = (self.data_w - 1) * 1 + self.sqe_kernel_size - self.data_w
        return int(np.ceil(h_pad / 2)), int(np.ceil(w_pad / 2))


class CurrentNet(nn.Module):
    def __init__(self, data_h, data_w, kernel_size, resnet_layers):
        super(CurrentNet, self).__init__()
        self.resnet_layers = resnet_layers
        self.cov1 = nn.Conv3d(2, 64, kernel_size=(1, 3, 3),
                              padding=(0, 1, 1), stride=(1, 1, 1))
        self.res = ResUnit(64, 64, data_h, data_w, kernel_size)
        self.cov2 = nn.Conv3d(64, 2, kernel_size=(1, 3, 3),
                              padding=(0, 1, 1), stride=(1, 1, 1))
        self.relu = nn.LeakyReLU(inplace=True)
        self.bn = nn.BatchNorm3d(64)

    def forward(self, inputs):
        output = self.cov1(inputs)
        output = self.relu(output)
        output = self.bn(output)
        output = self.cov2(output)
        return output


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


class FusionNet(nn.Module):
    def __init__(self, data_h, data_w, use_ext):
        super(FusionNet, self).__init__()
        self.w_w = nn.Parameter(torch.randn(1))
        self.w_d = nn.Parameter(torch.randn(1))
        self.w_c = nn.Parameter(torch.randn(1))
        self.data_h = data_h
        self.data_w = data_w
        self.use_ext = use_ext
        if use_ext:
            self.cov = nn.Conv3d(2, 2, kernel_size=(4, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))
        else:
            self.cov = nn.Conv3d(2, 2, kernel_size=(3, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))
        self.bn = nn.BatchNorm3d(2)
        self.relu = nn.LeakyReLU(inplace=False)
        self.tanh = nn.Tanh()

    def forward(self, week, day, current, ext):
        if self.use_ext:
            inputs = torch.stack([week, day, current, ext], dim=2).view(week.shape[0], 2, 4, self.data_h, self.data_w)
        else:
            inputs = torch.stack([week, day, current], dim=2).view(week.shape[0], 2, 3, self.data_h, self.data_w)
        output = self.bn(inputs)
        output = self.cov(output)
        return self.tanh(output)


class TestModule(nn.Module):
    def __init__(self, wind_size=7*48, batch_size=2, sqe_rate=3, sqe_kernel_size=3, dila_rate_list=None, tcn_kernel_size=3, week_resnet_layers=5,
                 current_resnet_layer=10, week_in_channel=None, week_out_channel=None, res_kernel_size=3, data_h=32, data_w=32,
                 day_in_channel=None, day_out_channel=None, use_ext=True):
        super(TestModule, self).__init__()
        # parameter
        # global
        self.batch_size = batch_size
        self.wind_size = wind_size
        self.day_size = 48
        self.heads = 1
        self.use_ext = use_ext
        # CovBlockAttentionNet
        self.sqe_rate = sqe_rate
        self.sqe_kernel_size = sqe_kernel_size
        # Tcn
        self.week_in_channel = None
        self.week_out_channel = None
        self.day_in_channel = None
        self.day_out_channel = None
        self.set_channel(week_in_channel, week_out_channel, day_in_channel, day_out_channel)
        self.dila_stride = 1  # static
        self.dila_rate_list = dila_rate_list
        self.tcn_kernel_size = tcn_kernel_size
        self.res_kernel_size = res_kernel_size
        self.data_h = data_h
        self.data_w = data_w
        # Resnet
        self.week_resnet_layers = week_resnet_layers
        self.current_resnet_layer = current_resnet_layer
        self.Week_SEN_Net = CovBlockAttentionNet(self.wind_size, self.sqe_rate, self.sqe_kernel_size, self.data_h, self.data_w)
        self.Day_SEN_Net = CovBlockAttentionNet(self.day_size, self.sqe_rate, self.sqe_kernel_size, self.data_h, self.data_w)
        self.Week_Tempora_Net = TemporalConvNet(dila_rate_type='week', dila_stride=self.dila_stride, tcn_kernel_size=self.tcn_kernel_size,
                                                wind_size=self.wind_size, resnet_layers=self.week_resnet_layers, in_channel=self.week_in_channel, out_channel=self.week_out_channel,
                                                res_kernel_size=self.res_kernel_size, data_h=self.data_h, data_w=self.data_w)
        self.Day_Tempora_Net = TemporalConvNet(dila_rate_type='day', dila_stride=self.dila_stride, tcn_kernel_size=self.tcn_kernel_size,
                                               wind_size=self.day_size, resnet_layers=self.week_resnet_layers, in_channel=self.day_in_channel, out_channel=self.day_out_channel,
                                               res_kernel_size=self.res_kernel_size, data_h=self.data_h, data_w=self.data_w)
        self.Current_Net = CurrentNet(data_h=self.data_h, data_w=self.data_w, kernel_size=self.res_kernel_size,
                                      resnet_layers=self.current_resnet_layer)
        self.tanh = nn.Tanh()
        self.emb = ExtEmb(self.data_h, self.data_w)
        self.fusion = FusionNet(self.data_h, self.data_w, self.use_ext)

    def forward(self, inputs, ext):
        week_data, day_data, current_data = self.split_data(inputs)
        sened_week_data = self.Week_SEN_Net(week_data)
        sened_day_data = self.Day_SEN_Net(day_data)
        week_tempora_net_res = self.Week_Tempora_Net(sened_week_data)
        day_tempora_net_res = self.Day_Tempora_Net(sened_day_data)
        current_net_res = self.Current_Net(current_data)
        if self.use_ext:
            ext = self.emb(ext)
        result = self.fusion(week_tempora_net_res, day_tempora_net_res, current_net_res, ext)
        return result

    def split_data(self, data):
        trend_data = data
        day_data = data[:, :, self.wind_size - self.day_size - 1:-1, :, :]
        current_data = data[:, :, 335:336, :, :]
        return trend_data, day_data, current_data

    def set_channel(self, week_in_channel, week_out_channel, day_in_channel, day_out_channel):
        if week_in_channel is None:
            self.week_in_channel = [2, 4, 8, 16, 32, 64, 128, 128]
        else:
            self.week_in_channel = week_in_channel
        if week_out_channel is None:
            self.week_out_channel = [4, 8, 16, 32, 64, 128, 128, 2]
        else:
            self.week_out_channel = week_out_channel
        if day_in_channel is None:
            self.day_in_channel = [2, 32, 32, 32, 32]
        else:
            self.day_in_channel = day_in_channel
        if day_out_channel is None:
            self.day_out_channel = [32, 32, 32, 32, 2]
        else:
            self.day_out_channel = day_out_channel