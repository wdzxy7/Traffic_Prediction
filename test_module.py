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
        self.resnet = nn.ModuleList()

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
    def __init__(self, dila_rate_type='week', dila_stride=1, tcn_kernel_size=3, wind_size=7*48, resnet_layers=5,
                 in_channel=None, out_channel=None, res_kernel_size=3, data_h=32, data_w=32):
        super(TemporalConvNet, self).__init__()
        self.dila_rate_type = dila_rate_type
        if dila_rate_type == 'week':
            dila_rate_list = [1, 1, 2, 4, 8, 8, 16, 32, 32, 64]
        else:
            dila_rate_list = [1, 1, 2, 2, 4, 8, 8]
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
        self.relu = nn.LeakyReLU(inplace=True)
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
        self.resUnit = ResUnit(2, 2, data_h, data_w, kernel_size)

    def forward(self, inputs):
        for i in range(self.resnet_layers):
            output = self.resUnit(inputs)
        return output


class FusionNet(nn.Module):
    def __init__(self, data_h, data_w):
        super(FusionNet, self).__init__()
        self.w_w = nn.Parameter(torch.randn(1))
        self.w_d = nn.Parameter(torch.randn(1))
        self.w_c = nn.Parameter(torch.randn(1))
        self.data_h = data_h
        self.data_w = data_w
        self.cov = nn.Conv3d(2, 2, kernel_size=(3, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))
        self.bn = nn.BatchNorm3d(2)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, week, day, current):
        # return self.fusion_2(week, day, current)
        inputs = torch.stack([week, day, current], dim=2).view(week.shape[0], 2, 3, self.data_h, self.data_w)
        output = self.bn(inputs)
        output = self.cov(output)
        return output

    def fusion_2(self, week, day, current):
        out = torch.add(self.w_w * week, self.w_d * day)
        out = torch.add(out, self.w_c * current)
        return out


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(-1)

    def forward(self, Q, K):
        d_k = K.size(-1)
        outputs = torch.matmul(Q, torch.transpose(K, -1, -2))
        outputs = (outputs / d_k)
        outputs = self.softmax(outputs)
        return outputs


class MultiAttention(nn.Module):
    def __init__(self, data_h, data_w, heads):
        super(MultiAttention, self).__init__()
        self.heads = heads
        self.data_h = data_h
        self.data_w = data_w
        self.input_size = data_h * data_w * 2 + 28
        self.output_size = (data_h * data_w * 2 + 28) * heads
        self.ext_seq = nn.Sequential(nn.Conv2d(336, 128, (1, 1), bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Conv2d(128, 64, (1, 1), bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Conv2d(64, 32, (1, 1), bias=False),
                                     nn.BatchNorm2d(32),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Conv2d(32, 16, (1, 1), bias=False),
                                     nn.BatchNorm2d(16),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Conv2d(16, 1, (1, 1), bias=False),
                                     nn.BatchNorm2d(1),
                                     nn.LeakyReLU(inplace=True),
                                     )
        self.linear1 = nn.Linear(self.input_size, self.output_size, bias=False)
        self.linear2 = nn.Linear(self.input_size, self.output_size, bias=False)
        self.linear3 = nn.Linear(self.input_size, self.output_size, bias=False)
        self.attention = Attention()
        self.linear_out = nn.Linear(self.output_size, 1, bias=False)

    def forward(self, inputs, ext):
        ext = ext.unsqueeze(-1).unsqueeze(-1).view(inputs.shape[0], 336, 28, 1)
        ext = self.ext_seq(ext)
        ext = ext.view(inputs.shape[0], 28)
        inputs = inputs.view(inputs.shape[0], self.data_h * self.data_w * 2)
        inputs_ = inputs
        attention_data = torch.cat([inputs, ext], dim=1)
        q = self.linear1(attention_data)
        k = self.linear2(attention_data)
        v = self.linear3(attention_data)
        outputs = self.attention(q, k)
        outputs = torch.matmul(outputs, v)
        outputs = outputs[:, 0:2048]
        outputs = outputs.view(inputs.shape[0], 2, 1, self.data_h, self.data_w)
        return outputs


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
        self.fusion = FusionNet(self.data_h, self.data_w)
        self.att = MultiAttention(self.data_h, self.data_w, self.heads)

    def forward(self, inputs, ext):
        week_data, day_data, current_data = self.split_data(inputs)
        sened_week_data = self.Week_SEN_Net(week_data)
        sened_day_data = self.Day_SEN_Net(day_data)
        week_tempora_net_res = self.Week_Tempora_Net(sened_week_data)
        day_tempora_net_res = self.Day_Tempora_Net(sened_day_data)
        current_net_res = self.Current_Net(current_data)
        fusion_res = self.fusion(week_tempora_net_res, day_tempora_net_res, current_net_res)
        if self.use_ext:
            result = self.att(fusion_res, ext)
        else:
            result = fusion_res
        return result

    def split_data(self, data):
        trend_data = data
        day_data = data[:, :, self.wind_size - self.day_size - 1:-1, :, :]
        current_data = data[:, :, 335:336, :, :]
        return trend_data, day_data, current_data

    def set_channel(self, week_in_channel, week_out_channel, day_in_channel, day_out_channel):
        if week_in_channel is None:
            self.week_in_channel = [2, 4, 8, 16, 32, 64, 32, 16, 8, 4]
        else:
            self.week_in_channel = week_in_channel
        if week_out_channel is None:
            self.week_out_channel = [4, 8, 16, 32, 64, 32, 16, 8, 4, 2]
        else:
            self.week_out_channel = week_out_channel
        if day_in_channel is None:
            self.day_in_channel = [2, 4, 8, 16, 16, 8, 4]
        else:
            self.day_in_channel = day_in_channel
        if day_out_channel is None:
            self.day_out_channel = [4, 8, 16, 16, 8, 4, 2]
        else:
            self.day_out_channel = day_out_channel
