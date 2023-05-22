"""
@author: 
"""

"""
This code will construct the Vandermonde matrices given the positions. 
"""

import torch
import numpy as np
import pdb

# class for fully nonequispaced 2d points
class VFT:
    def __init__(self, x_positions, y_positions, modes):
        # it is important that positions are scaled between 0 and 2*pi
        x_positions -= torch.min(x_positions)
        self.x_positions = x_positions * 6.28 / torch.max(x_positions)
        y_positions -= torch.min(y_positions)
        self.y_positions = y_positions * 6.28 / torch.max(y_positions)
        self.number_points = x_positions.shape[1]
        self.batch_size = x_positions.shape[0]
        self.modes = modes

        self.X = torch.arange(modes).repeat(self.batch_size, 1)[:,:,None].float().cuda()

        self.V_fwd, self.V_inv = self.make_matrix()

    def make_matrix(self):
        forward_mat_ = torch.zeros((self.batch_size, self.modes**2, self.number_points), dtype=torch.cfloat).cuda()
        for Y in range(self.modes):
            Y_mat = Y*self.y_positions[:,None,:].expand(-1, self.modes, -1)
            X_mat = torch.bmm(self.X, self.x_positions[:,None,:])
            forward_mat_[:, Y*self.modes:(Y+1)*self.modes, :] = torch.exp(-1j* (X_mat+Y_mat))
        forward_mat = forward_mat_ / np.sqrt(self.number_points) * np.sqrt(2)

        inverse_mat = torch.conj(forward_mat).permute(0,2,1)

        return forward_mat, inverse_mat

    def forward(self, data):
        data_fwd = torch.bmm(self.V_fwd, data)
        return data_fwd

    def inverse(self, data):
        data_inv = torch.bmm(self.V_inv, data)
        
        return data_inv