"""
@author: 
"""

"""
This is the FNO class (Li et al., Fourier Neural Operator for Parametric Partial Differential Equations, https://arxiv.org/abs/2010.08895)
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


################################################################
# fourier layer
################################################################

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))


    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.fft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.ifft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2D(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2D, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 0 # pad the domain if input is non-periodic

        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.fc0 = nn.Linear(3, self.width).to(torch.cfloat)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
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

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x.real) + 1j * F.gelu(x.imag)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        # gridx = x_pos / torch.max(x_pos)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, size_y, 1, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        # gridy = y_pos / torch.max(y_pos)
        gridy = gridy.reshape(1, size_y, 1, 1).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)