"""
@author: 
"""

"""
This class will load the data for the airfoil problem from the paper from Li et al., Fourier Neural Operator
with Learned Deformations for PDEs on General Geometries, which can be found here: https://arxiv.org/abs/2207.05209.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import socket
import pdb

class LoadCylinder():
    def __init__(self, training_samples, testing_samples, file_path=''):
        self.file_path = file_path

        self.Q = np.load(f'{file_path}NACA_Cylinder_Q.npy', allow_pickle=True)
        self.X = np.load(f'{file_path}NACA_Cylinder_X.npy', allow_pickle=True)
        self.Y = np.load(f'{file_path}NACA_Cylinder_Y.npy', allow_pickle=True)
        
        self.training_samples = training_samples
        self.testing_samples = testing_samples


    def get_data(self, range_norm=True, small_domain=True):
        # given:    the number of samples
        # return:   the inputs (the grid) and the outputs (the pressure) for the airfoil
        
        num_samples = self.training_samples + self.testing_samples

        x_tensor = torch.tensor(self.X[:num_samples]).to(torch.float)
        y_tensor = torch.tensor(self.Y[:num_samples]).to(torch.float)
        output = torch.tensor(self.Q[:num_samples]).to(torch.float)

        if small_domain:
            width = 45
            depth = 8
            x_tensor = x_tensor[:, depth:-depth, :width]
            y_tensor = y_tensor[:, depth:-depth, :width]
            output = output[:, 0, depth:-depth, :width]
        else:
            output = output[:,0,:,:]


        x_tensor = torch.flatten(x_tensor, start_dim=1, end_dim=2)
        y_tensor = torch.flatten(y_tensor, start_dim=1, end_dim=2)
        output = torch.flatten(output, start_dim=1, end_dim=2)

        input = torch.cat((x_tensor.unsqueeze(-1), y_tensor.unsqueeze(-1)), -1)
        

        if range_norm:
            output = regularize(output)

        return input, output

    def regularize(self, field):
        field -= torch.min(field)
        field /= torch.max(field)
        return field
    
    def get_dataloaders(self, batch_size, range_norm=False, small_domain=True):
        
        inputs, outputs = self.get_data(range_norm=range_norm, small_domain=small_domain)
        train_in = inputs[:self.training_samples]
        train_out = outputs[:self.training_samples]
        test_in = inputs[-self.testing_samples:]
        test_out = outputs[-self.testing_samples:]
        
        train_loader = DataLoader(TensorDataset(train_in, train_out), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_in, test_out), batch_size=1, shuffle=False)

        return train_loader, test_loader
