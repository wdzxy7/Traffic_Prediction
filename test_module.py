import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F


class CausalConvolution(nn.Module):
    def __int__(self):
        super(CausalConvolution, self).__init__()
        pass

    def forward(self, inputs):
        pass


class ResUnit(nn.Module):
    def __int__(self):
        super(ResUnit, self).__init__()
        pass

    def forward(self, inputs):
        pass


class SeqAndExcNet(nn.Module):
    def __int__(self, wind_size, r):
        super(SeqAndExcNet, self).__init__()
        mid_layer = np.ceil(wind_size / r)
        self.squeeze_layer = nn.AdaptiveAvgPool3d((wind_size, 1, 1))
        self.linear1 = nn.Linear(wind_size, mid_layer)
        self.linear2 = nn.Linear(mid_layer, wind_size)

    def forward(self, inputs):
        output = self.squeeze_layer(inputs)
        output = self.linear1(output)
        output = nn.ReLU(output)
        output = self.linear2(output)
        output = F.sigmoid(output)
        return torch.bmm(inputs, output)


class TestModule(nn.Module):
    def __int__(self):
        super(TestModule, self).__init__()
        pass

    def forward(self, inputs):
        pass