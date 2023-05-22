"""
@author: 
"""

"""
This class will load the data for the airfoil problem from the paper from Li et al., Fourier Neural Operator
with Learned Deformations for PDEs on General Geometries, which can be found here: https://arxiv.org/abs/2207.05209.
"""
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import socket
import pdb

class LoadAirfoil():
    def __init__(self, training_samples, testing_samples, file_path=''):
        self.file_path = file_path

        self.Q = np.load(f'{file_path}NACA_Q.npy', allow_pickle=True)
        self.X = np.load(f'{file_path}NACA_X.npy', allow_pickle=True)
        self.Y = np.load(f'{file_path}NACA_Y.npy', allow_pickle=True)
        
        self.training_samples = training_samples
        self.testing_samples = testing_samples

    def get_data(self, range_norm=True):
        # given:    the number of samples
        # return:   the inputs (the grid) and the outputs (the pressure) for the airfoil
        
        num_samples = self.training_samples + self.testing_samples

        x_tensor = torch.tensor(self.X).unsqueeze(-1).to(torch.float)
        y_tensor = torch.tensor(self.Y).unsqueeze(-1).to(torch.float)

        input = torch.cat((x_tensor[:num_samples], y_tensor[:num_samples]), -1)
        output = torch.tensor(self.Q[:num_samples, :1,:]).unsqueeze(-1).to(torch.float)

        if range_norm:
            output = regularize(output)

        return input, output

    def regularize(self, field):
        field -= torch.min(field)
        field /= torch.max(field)
        return field
    
    def get_dataloaders(self, batch_size, range_norm=True):
        
        inputs, outputs = self.get_data(range_norm=range_norm)
        train_in = inputs[:self.training_samples]
        train_out = outputs[:self.training_samples]
        test_in = inputs[-self.testing_samples:]
        test_out = outputs[-self.testing_samples:]
        
        train_loader = DataLoader(TensorDataset(train_in, train_out), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_in, test_out), batch_size=batch_size, shuffle=True)

        return train_loader, test_loader
