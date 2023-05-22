import torch
import numpy as np
import pdb

# class for 1-dimensional Fourier transforms on nonequispaced data, using the adjoint as an approximate inverse
class VNO1D:
    def __init__(self, positions, modes):
        self.modes = modes
        self.positions = positions / torch.max(positions) * 2 * np.pi
        self.l = positions.shape[0]

        self.Vt, self.Vc = self.make_matrix()

    def make_matrix(self):
        V = torch.zeros([self.modes, self.l], dtype=torch.cfloat).cuda()
        for row in range(self.modes):
            V[row,:] = np.exp(-1j * row * self.positions)
        V = torch.divide(V, np.sqrt(self.l)) * np.sqrt(2)

        return torch.transpose(V, 0, 1), torch.conj(V)

    def forward(self, data):
        return torch.matmul(data, self.Vt)

    def inverse(self, data):
        return torch.matmul(data, self.Vc)
