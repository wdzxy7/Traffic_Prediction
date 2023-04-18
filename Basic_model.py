import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable


class ExtEmb(nn.Module):
    def __init__(self, data_h, data_w, ext_dim):
        super(ExtEmb, self).__init__()
        self.data_h = data_h
        self.data_w = data_w
        self.emb = nn.Embedding(num_embeddings=4, embedding_dim=1)
        self.linear1 = nn.Linear(4 * ext_dim, 28)
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


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list[-1]

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvLSTMNet(nn.Module):
    def __init__(self, data_h=32, data_w=32, use_ext=True, trend_len=7, current_len=4, ext_dim=28):
        super(ConvLSTMNet, self).__init__()
        # parameter
        # global
        self.heads = 1
        self.use_ext = use_ext
        self.trend_len = trend_len
        self.current_lend = current_len
        # CovBlockAttentionNet
        # Tcn
        self.data_h = data_h
        self.data_w = data_w
        # Resnet
        self.tanh = nn.Tanh()
        self.emb = ExtEmb(self.data_h, self.data_w, ext_dim)
        c_in = (trend_len * 2 + current_len + 1) * 2
        self.up_channel = nn.Sequential(nn.Conv2d(c_in, 64, 1, 1),
                                        nn.BatchNorm2d(64),
                                        nn.LeakyReLU(inplace=True))

        self.conv_lstm = ConvLSTM(input_dim=64,
                                  hidden_dim=[64, 64, 64, 64, 64, 64, 64, 2],
                                  kernel_size=(3, 3),
                                  num_layers=8,
                                  batch_first=True,
                                  bias=True,
                                  return_all_layers=False)

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
        output = output.view(output.shape[0], 1, output.shape[1], output.shape[2], output.shape[3])
        output = self.conv_lstm(output)
        output = output.view(output.shape[0], output.shape[2], output.shape[3], output.shape[4])
        output = self.tanh(output)
        # output = self.Out_Net(output)
        return output.view(inputs.shape[0], 2, 1, self.data_h, self.data_w)

    def merge_data(self, data, ext):
        current_data = []
        leak_data = []
        day_data = []
        T = 48
        for i in range(1, self.trend_len + 1):
            leak_data.append(data[:, :, 336 - i * T:337 - i * T, :, :])
            day_data.append(data[:, :, 335 - (i - 1) * T:336 - (i - 1) * T, :, :])
            if i < self.current_lend + 1:
                current_data.append(data[:, :, 336 - i:337 - i, :, :])
        leak_data.append(ext)
        data = torch.stack(current_data + day_data + leak_data, dim=1)
        shape = data.shape
        return data.view(shape[0], shape[1] * shape[2], shape[4], shape[5])


class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        init.orthogonal(self.reset_gate.weight)
        init.orthogonal(self.update_gate.weight)
        init.orthogonal(self.out_gate.weight)
        init.constant(self.reset_gate.bias, 0.)
        init.constant(self.update_gate.bias, 0.)
        init.constant(self.out_gate.bias, 0.)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = Variable(torch.zeros(state_size)).cuda()
            else:
                prev_state = Variable(torch.zeros(state_size))

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = F.sigmoid(self.update_gate(stacked_inputs))
        reset = F.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = F.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class ConvGRU(nn.Module):

    def __init__(self, input_size, hidden_sizes, kernel_sizes, n_layers):

        super(ConvGRU, self).__init__()

        self.input_size = input_size

        if type(hidden_sizes) != list:
            self.hidden_sizes = [hidden_sizes]*n_layers
        else:
            assert len(hidden_sizes) == n_layers, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_sizes = hidden_sizes
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes]*n_layers
        else:
            assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers

        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]

            cell = ConvGRUCell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i])
            name = 'ConvGRUCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells


    def forward(self, x, hidden=None):
        if not hidden:
            hidden = [None]*self.n_layers

        input_ = x

        upd_hidden = []

        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = hidden[layer_idx]

            # pass through layer
            upd_cell_hidden = cell(input_, cell_hidden)
            upd_hidden.append(upd_cell_hidden)
            # update input_ to the last updated hidden layer for next pass
            input_ = upd_cell_hidden

        # retain tensors in list to allow different hidden sizes
        return upd_hidden[-1]


class GRUNet(nn.Module):
    def __init__(self, data_h=32, data_w=32, use_ext=True, trend_len=7, current_len=4, ext_dim=28):
        super(GRUNet, self).__init__()
        # parameter
        # global
        self.heads = 1
        self.use_ext = use_ext
        self.trend_len = trend_len
        self.current_lend = current_len
        # CovBlockAttentionNet
        # Tcn
        self.data_h = data_h
        self.data_w = data_w
        # Resnet
        self.tanh = nn.Tanh()
        self.emb = ExtEmb(self.data_h, self.data_w, ext_dim)
        c_in = (trend_len * 2 + current_len + 1) * 2
        self.up_channel = nn.Sequential(nn.Conv2d(c_in, 32, 1, 1),
                                        nn.BatchNorm2d(32),
                                        nn.LeakyReLU(inplace=True))

        self.gru = ConvGRU(input_size=32,
                           hidden_sizes=[32, 32, 32, 32, 2],
                           kernel_sizes=3,
                           n_layers=5)

    def forward(self, inputs, ext):
        if self.use_ext:
            ext = self.emb(ext)
        inputs = self.merge_data(inputs, ext)
        output = self.up_channel(inputs)
        output = output
        output = self.gru(output)
        return output.view(inputs.shape[0], 2, 1, self.data_h, self.data_w)

    def merge_data(self, data, ext):
        current_data = []
        leak_data = []
        day_data = []
        T = 48
        for i in range(1, self.trend_len + 1):
            leak_data.append(data[:, :, 336 - i * T:337 - i * T, :, :])
            day_data.append(data[:, :, 335 - (i - 1) * T:336 - (i - 1) * T, :, :])
            if i < self.current_lend + 1:
                current_data.append(data[:, :, 336 - i:337 - i, :, :])
        leak_data.append(ext)
        data = torch.stack(current_data + day_data + leak_data, dim=1)
        shape = data.shape
        return data.view(shape[0], shape[1] * shape[2], shape[4], shape[5])