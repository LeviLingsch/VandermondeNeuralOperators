"""
@author: 
"""

"""
This class constructs the VNO model.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, transformer):
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

        self.transformer = transformer

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = self.transformer.forward(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  2*self.modes1, self.modes2, dtype=torch.cfloat, device=x.device)
        # out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft, self.weights1)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = self.transformer.inverse(out_ft)

        return x
    

class VNO2DFixed(nn.Module):
    def __init__(self, modes1, modes2, width, transformer, sparse_x):
        super(VNO2DFixed, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.x = sparse_x
        self.padding = 0 # pad the domain if input is non-periodic
        
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.fc0 = nn.Linear(3, self.width).to(torch.cfloat)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, transformer)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, transformer)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, transformer)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, transformer)
        self.w0r = nn.Conv2d(self.width, self.width, 1)
        self.w1r = nn.Conv2d(self.width, self.width, 1)
        self.w2r = nn.Conv2d(self.width, self.width, 1)
        self.w3r = nn.Conv2d(self.width, self.width, 1)
        self.w0i = nn.Conv2d(self.width, self.width, 1)
        self.w1i = nn.Conv2d(self.width, self.width, 1)
        self.w2i = nn.Conv2d(self.width, self.width, 1)
        self.w3i = nn.Conv2d(self.width, self.width, 1)
        self.fc1 = nn.Linear(self.width, 128).to(torch.cfloat)
        self.fc2 = nn.Linear(128, 1).to(torch.cfloat)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0r(x.real) + 1j * self.w0i(x.imag)
        x = x1 + x2
        x = F.gelu(x.real) + 1j * F.gelu(x.imag)

        x1 = self.conv1(x)
        x2 = self.w1r(x.real) + 1j * self.w1i(x.imag)
        x = x1 + x2
        x = F.gelu(x.real) + 1j * F.gelu(x.imag)

        x1 = self.conv2(x)
        x2 = self.w2r(x.real) + 1j * self.w2i(x.imag)
        x = x1 + x2
        x = F.gelu(x.real) + 1j * F.gelu(x.imag)

        x1 = self.conv3(x)
        x2 = self.w3r(x.real) + 1j * self.w3i(x.imag)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x.real) + 1j * F.gelu(x.imag)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[1]
        gridx = self.x / torch.max(self.x)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, size_y, 1, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, size_y, 1, 1).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
