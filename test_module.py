import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    def __init__(self, dila_rate, dila_size):
        super(TemporalBlock, self).__init__()
        self.dila_rate = dila_rate
        self.dila_size = dila_size

    def forward(self):
        pass

    def re_sort_data(self, data):
        pass


class TemporalConvNet(nn.Module):
    def __init__(self, causal_rate, dila_rate, causal_layer):
        super(TemporalConvNet, self).__init__()
        self.causal_rate = causal_rate
        self.dila_rate = dila_rate
        self.causal_layer = causal_layer
        self.dila_size = 1  # dila size
        self.layer_list = []
        self.in_channel = None
        self.out_channel = None
        for i in range(self.causal_layer):
            self.dila_size = 2 ** self.dila_size
            if self.dila_size > 8:
                self.dila_size = 8
            self.layer_list.append(TemporalBlock(self.dila_rate, self.dila_size))

    def forward(self, inputs):
        pass

    def cal_channel(self):
        pass


class AvgResUnit(nn.Module):
    def __init__(self):
        super(AvgResUnit, self).__init__()
        pass

    def forward(self, inputs):
        pass


class SeqAndExcNet(nn.Module):
    def __init__(self, wind_size, batch_size, r):
        super(SeqAndExcNet, self).__init__()
        self.wind_size = wind_size
        self.batch_size = batch_size
        mid_layer = int(np.ceil(self.wind_size / r))
        self.squeeze_layer = nn.AdaptiveAvgPool3d((wind_size, 1, 1))
        self.linear1 = nn.Linear(wind_size, mid_layer)
        self.linear2 = nn.Linear(mid_layer, wind_size)

    def forward(self, inputs):
        output = self.squeeze_layer(inputs)
        output = output.view(self.batch_size, 2, self.wind_size)
        output1, output2 = output.chunk(2, 1)
        outputs = [output1, output2]
        res_out = []
        for i in range(0, 2):
            c_out = self.linear1(outputs[i])
            c_out = F.relu(c_out)
            c_out = self.linear2(c_out)
            c_out = F.sigmoid(c_out)
            res_out.append(c_out)
        res_out = torch.stack(res_out, dim=1).view(self.batch_size, 2, self.wind_size, 1, 1)
        return torch.mul(inputs, res_out)


class TestModule(nn.Module):
    def __init__(self, wind_size=7*48, batch_size=2, sqe_rate=3, dila_function='square', causal_rate=3, causal_layer=8,
                 kernel_size=3):
        super(TestModule, self).__init__()
        self.dila_function = dila_function  # two function add or square
        self.dila_rate = 1  # static
        self.causal_rate = causal_rate  # cov channel sum
        self.causal_layer = causal_layer  # layer of causal network
        self.batch_size = batch_size
        self.wind_size = wind_size
        self.sqe_rate = sqe_rate
        self.kernel_size = kernel_size
        self.SEN_Net = SeqAndExcNet(self.wind_size, self.batch_size, self.sqe_rate)
        self.Tempora_Net = TemporalConvNet(self.causal_rate, self.dila_rate, self.dila_function, self.causal_layer)

    def forward(self, inputs):
        x = inputs[0]
        y = inputs[1]
        ext = inputs[2]
        sened_x = self.SEN_Net(x)