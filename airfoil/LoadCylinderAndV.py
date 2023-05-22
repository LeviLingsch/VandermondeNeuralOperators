"""
@author:
"""

"""
This class will load the data for the airfoil problem and precompute the Vandermonde matrices. The data is taken
from the paper from Li et al., Fourier Neural Operator with Learned Deformations for PDEs on General Geometries, 
which can be found here: https://arxiv.org/abs/2207.05209.
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import socket
import pdb

class LoadCylinderAndV():
    def __init__(self, training_samples, testing_samples, file_path=''):
        self.file_path = file_path

        self.Q = np.load(f'{file_path}NACA_Cylinder_Q.npy', allow_pickle=True)
        self.X = np.load(f'{file_path}NACA_Cylinder_X.npy', allow_pickle=True)
        self.Y = np.load(f'{file_path}NACA_Cylinder_Y.npy', allow_pickle=True)
        
        self.training_samples = training_samples
        self.testing_samples = testing_samples

    def make_matrix(self, modes, x_positions, y_positions):
        # given:    the modes and the x/y positions
        # return:   the matrices to compute the Fourier transformation

        number_samples = x_positions.shape[0]
        number_points = x_positions.shape[1]

        forward_mat = torch.zeros((number_samples, modes**2, number_points), dtype=torch.cfloat)
        for Y in range(modes):
            for X in range(modes):
                forward_mat[:, Y+X*modes, :] = torch.exp(-1j* (X*x_positions+Y*y_positions))
        forward_mat = forward_mat / np.sqrt(number_points) * np.sqrt(2)

        inverse_mat = torch.zeros((number_samples, number_points, modes**2),  dtype=torch.cfloat)
        for Y in range(modes):
            for X in range(modes):
                inverse_mat[:, :, Y+X*modes] = torch.exp(1j* (X*x_positions+Y*y_positions))
        inverse_mat = inverse_mat  / np.sqrt(number_points) * np.sqrt(2)

        return forward_mat, inverse_mat

    def get_data(self, modes, range_norm=False, small_domain=True):
        # given:    the number of samples
        # return:   the inputs (the grid) and the outputs (the pressure) for the airfoil
        
        num_samples = self.training_samples + self.testing_samples

        x_tensor = torch.tensor(self.X[:num_samples]).to(torch.float)
        y_tensor = torch.tensor(self.Y[:num_samples]).to(torch.float)
        output = torch.tensor(self.Q[:num_samples]).to(torch.float)

        if small_domain:
            x_tensor = x_tensor[:, 8:-8, :45]
            y_tensor = y_tensor[:, 8:-8, :45]
            output = output[:, 0, 8:-8, :45]

        x_tensor = torch.flatten(x_tensor, start_dim=1, end_dim=2)
        y_tensor = torch.flatten(y_tensor, start_dim=1, end_dim=2)
        
        V_fwd, V_inv = self.make_matrix(modes, x_tensor, y_tensor)
        output = torch.flatten(output, start_dim=1, end_dim=2)

        input = torch.cat((x_tensor.unsqueeze(-1), y_tensor.unsqueeze(-1)), -1)
        

        if range_norm:
            output = regularize(output)

        return input, output, V_fwd, V_inv

    def regularize(self, field):
        field -= torch.min(field)
        field /= torch.max(field)
        return field
    
    def get_dataloaders(self, batch_size, modes, range_norm=False, small_domain=True):
        
        inputs, outputs, vfwd, vinv = self.get_data(modes, range_norm=range_norm, small_domain=small_domain)
        train_in = inputs[:self.training_samples]
        train_out = outputs[:self.training_samples]
        train_fwd = vfwd[:self.training_samples]
        train_inv = vinv[:self.training_samples]

        test_in = inputs[-self.testing_samples:]
        test_out = outputs[-self.testing_samples:]
        test_fwd = vfwd[-self.testing_samples:]
        test_inv = vinv[-self.testing_samples:]
        
        
        train_loader = DataLoader(TensorDataset(train_in, train_out, train_fwd, train_inv), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_in, test_out, test_fwd, test_inv), batch_size=batch_size, shuffle=False)

        return train_loader, test_loader
