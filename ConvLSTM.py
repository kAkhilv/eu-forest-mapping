import torch
import torch.nn as nn
class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.i_dim = input_dim
        self.h_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.i_dim + self.h_dim,
            out_channels=4 * self.h_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.h_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(torch.nn.Module):
    def __init__(self, height, width, input_dim=13, hidden_dim=16, nclasses=4, kernel_size=(3, 3), bias=False):
        super(ConvLSTM, self).__init__()

        self.inconv = torch.nn.Conv3d(input_dim, hidden_dim, (1, 3, 3))

        self.cell = ConvLSTMCell(
            input_size=(height, width), input_dim=hidden_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, bias=bias
        )

        self.final = torch.nn.Conv2d(hidden_dim, nclasses, (3, 3))

    def forward(self, x, hidden=None, state=None):
        x = x.permute(0, 4, 1, 2, 3)
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), "constant", 0)
        x = self.inconv.forward(x)

        b, c, t, h, w = x.shape
        if hidden is None:
            hidden = torch.zeros((b, c, h, w))
        if state is None:
            state = torch.zeros((b, c, h, w))

        if torch.cuda.is_available():
            hidden = hidden.cuda()
            state = state.cuda()

        for iter in range(t):
            hidden, state = self.cell.forward(x[:, :, iter, :, :], (hidden, state))

        x = torch.nn.functional.pad(state, (1, 1, 1, 1), "constant", 0)
        x = self.final.forward(x)

        return x