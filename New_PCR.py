import torch
import numpy as np
import torch.nn as nn
from humanfriendly.terminal import output

'''
Parallel Hybrid Residual Networks
'''

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
    def __init__(self, data_h, data_w, ext_dim):
        super(ExtEmb, self).__init__()
        self.data_h = data_h
        self.data_w = data_w
        self.emb = nn.Embedding(num_embeddings=4, embedding_dim=1)
        self.linear1 = nn.Linear(4 * ext_dim, 128)
        self.relu = nn.LeakyReLU(inplace=False)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 2 * self.data_h * self.data_w)
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
        output = output.view(shape[0], 2, self.data_h, self.data_w)
        return output


class Attention(nn.Module):
    def __init__(self, dim, att_dim):
        super(Attention, self).__init__()
        self.upper = nn.Linear(dim, att_dim)
        self.linear1 = nn.Linear(att_dim, att_dim)
        self.linear2 = nn.Linear(att_dim, att_dim)
        self.linear3 = nn.Linear(att_dim, att_dim)
        self.softmax = nn.LogSoftmax(-1)
        self.down = nn.Linear(att_dim, dim)

    def forward(self, time_feature):
        time_feature = torch.unsqueeze(time_feature, 2)
        time_feature = time_feature.transpose(2, 1)
        time_feature = self.upper(time_feature)
        q = self.linear1(time_feature)
        k = self.linear2(time_feature)
        v = self.linear3(time_feature)
        d_k = k.size(-1)
        qk = torch.bmm(q.transpose(2, 1), k)
        qk = qk / d_k
        att_value = self.softmax(qk)
        output = torch.bmm(v, att_value)
        output = self.down(output)
        return output.transpose(2, 1)


class PHRNet(nn.Module):
    def __init__(self, sqe_rate=3, sqe_kernel_size=3, resnet_layers=5, res_kernel_size=3, data_h=32, data_w=32,
                 use_ext=True, trend_len=7, current_len=4, ext_dim=28, nyc=False, conv_dim=64, att_dim=64):
        super(PHRNet, self).__init__()
        # parameter
        # global
        self.nyc = nyc
        self.heads = 1
        self.use_ext = use_ext
        self.trend_len = trend_len
        self.current_lend = current_len
        # CovBlockAttentionNet
        self.sqe_rate = sqe_rate
        self.sqe_kernel_size = sqe_kernel_size
        # Tcn
        self.data_h = data_h
        self.data_w = data_w
        self.att_layers = 3
        # Resnet
        self.att_dim = att_dim
        self.conv_dim = conv_dim
        self.resnet_layers = resnet_layers
        self.res_kernel_size = res_kernel_size
        self.tanh = nn.Tanh()
        self.ext_net = ExtEmb(self.data_h, self.data_w, ext_dim)
        self.c_in = (trend_len * 2 + current_len) * 2
        self.up_channel = nn.Sequential(nn.Conv2d(self.c_in, self.conv_dim, 1, 1),
                                        nn.BatchNorm2d(self.conv_dim),
                                        nn.LeakyReLU(inplace=True))
        self.Res_Net = self.build_resnet()
        self.Dila_Net = self.build_dila_resnet()
        # self.Sen_Net = CovBlockAttentionNet(64, self.sqe_rate, self.sqe_kernel_size, self.data_h, self.data_w)
        self.Out_Net = nn.Sequential(nn.Conv2d(self.conv_dim, 32, kernel_size=3, stride=1, padding=(1, 1)),
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
        self.Sut_Net = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=1, stride=1),
            nn.Tanh(),
            nn.Conv2d(2, 2, kernel_size=1, stride=1),
            nn.Tanh()
        )
        self.down = nn.AdaptiveAvgPool2d((1, 1))
        self.tanh = nn.Tanh()
        self.att, self.down_temporal = self.build_att()

    def forward(self, inputs, ext):
        if self.use_ext:
            ext = self.ext_net(ext)
        if self.nyc:
            inputs, diffs = self.merge_data2(inputs)
        else:
            inputs, diffs = self.merge_data(inputs)
        outputs = self.spatial_feature(inputs)
        temporal = self.temporal_feature(diffs)
        if self.use_ext:
            res = outputs + temporal + ext
        else:
            res = outputs + temporal
        res = self.Sut_Net(res)
        return torch.unsqueeze(res, 2)

    def spatial_feature(self, inputs):
        outputs = self.up_channel(inputs)
        res_out = outputs
        dila_out = outputs
        for i in range(self.resnet_layers):
            res_out = self.Res_Net[i](res_out)
            dila_out = self.Dila_Net[i](dila_out)
        outputs = res_out + dila_out
        outputs = self.Out_Net(outputs)
        return outputs

    def temporal_feature(self, inputs):
        temporal = inputs
        for i in range(self.att_layers):
            att_inputs = self.down(temporal)
            temporal = self.compute(temporal, att_inputs, self.att[i])
            temporal = self.down_temporal[i](temporal)
        return temporal

    def merge_data(self, data):
        current_data = []
        leak_data = []
        day_data = []
        diffs = []
        T = 48
        for i in range(1, self.trend_len + 1):
            leak_data.append(data[:, :, 336 - i * T:337 - i * T, :, :])
            diff = data[:, 0, 336 - i * T:337 - i * T, :, :] - data[:, 1, 336 - i * T:337 - i * T, :, :]
            diffs.append(diff)
            day_data.append(data[:, :, 335 - (i - 1) * T:336 - (i - 1) * T, :, :])
            diff = data[:, 0, 335 - (i - 1) * T:336 - (i - 1) * T, :, :] - data[:, 1, 335 - (i - 1) * T:336 - (i - 1) * T, :, :]
            diffs.append(diff)
            if i < self.current_lend + 1:
                current_data.append(data[:, :, 336 - i:337 - i, :, :])
                diff = data[:, 0, 336 - i:337 - i, :, :] - data[:, 1, 336 - i:337 - i, :, :]
                diffs.append(diff)
        all_data = current_data + day_data + leak_data
        data = torch.cat(all_data, dim=1)
        data = torch.squeeze(data)
        diffs = torch.cat(diffs, dim=1)
        diffs = torch.squeeze(diffs)
        return data, diffs

    # For BikeNYC
    def merge_data2(self, data):
        current_data = []
        leak_data = []
        day_data = []
        diffs = []
        T = 24
        for i in range(1, self.trend_len + 1):
            leak_data.append(data[:, :, 168 - i * T:169 - i * T, :, :])
            diff = data[:, 0, 168 - i * T:169 - i * T, :, :] - data[:, 1, 168 - i * T:169 - i * T, :, :]
            diffs.append(torch.unsqueeze(diff, dim=2))
            day_data.append(data[:, :, 167 - (i - 1) * T:168 - (i - 1) * T, :, :])
            diff = data[:, 0, 167 - (i - 1) * T:168 - (i - 1) * T, :, :] - data[:, 1, 167 - (i - 1) * T:168 - (i - 1) * T, :, :]
            diffs.append(torch.unsqueeze(diff, dim=2))

            if i < self.current_lend + 1:
                current_data.append(data[:, :, 168 - i:169 - i, :, :])
                diff = data[:, 0, 168 - i:169 - i, :, :] - data[:, 1, 168 - i:169 - i, :, :]
                diffs.append(torch.unsqueeze(diff, dim=2))
        all_data = current_data + day_data + leak_data
        data = torch.cat(all_data, dim=1)
        data = torch.squeeze(data)
        diffs = torch.cat(diffs, dim=1)
        diffs = torch.squeeze(diffs)
        return data, diffs

    def build_resnet(self):
        res_net = nn.ModuleList()
        for i in range(self.resnet_layers):
            res_net.append(ResUnit(64, 64, self.data_h, self.data_w, self.res_kernel_size))
        return res_net

    def build_dila_resnet(self):
        dila_net = nn.ModuleList()
        for i in range(self.resnet_layers):
            dila_net.append(DilaResUnit(64, 64, self.data_h, self.data_w, self.res_kernel_size))
        return dila_net

    def build_att(self):
        att_net = nn.ModuleList()
        down_net = nn.ModuleList()
        for i in range(self.att_layers):
            sub_net = nn.Sequential(nn.Conv2d(self.c_in // 2, self.c_in // 4, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(self.c_in // 4),
                                    nn.LeakyReLU(inplace=True),
                                    )
            down_net.append(sub_net)
            att_net.append(Attention(self.c_in // 2, self.c_in // 2 * 4))
            self.c_in = self.c_in // 2
        return att_net, down_net

    def compute(self, base, temporal, att_net):
        temporal = torch.squeeze(temporal)
        att = att_net(temporal)
        original_shape = base.shape
        base = base.view(original_shape[0], original_shape[1], -1)
        output = base * att
        return self.tanh(output.reshape(original_shape))