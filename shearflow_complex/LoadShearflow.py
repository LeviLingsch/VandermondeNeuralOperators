"""
@author: 
"""

"""
This code will load the data for the Shear Layer problem. The x-velocity is loaded as a real value,
the y-velocity as an imaginary value, both within a complex number.
"""

import netCDF4
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import pdb


class LoadShearflow():
    def __init__(self, training_samples, testing_samples, file=''):
        self.in_size = 1024
        self.file = file
        self.ntrain = training_samples
        self.ntest = testing_samples
        training_inputs, training_outputs = self.get_data(training_samples+testing_samples)
        
        training_inputs = training_inputs.permute(0, 2, 3, 1)
        training_outputs = training_outputs.permute(0, 2, 3, 1)

        training_inputs  = self.normalize(training_inputs)
        training_outputs = self.normalize(training_outputs)
        
        testing_inputs = training_inputs[-testing_samples:] 
        testing_outputs = training_outputs[-testing_samples:] 
        training_inputs = training_inputs[:training_samples]
        training_outputs = training_outputs[:training_samples]

        self.testing_inputs = testing_inputs
        self.testing_outputs = testing_outputs
        self.training_inputs = training_inputs
        self.training_outputs = training_outputs

        
    def return_data(self):
        return self.training_inputs, self.training_outputs, self.testing_inputs, self.testing_outputs

    def normalize(self, data):
        m = torch.max(data.real)
        M = torch.min(data.real)
        real_data = (data.real - m)/(M - m)

        m = torch.max(data.imag)
        M = torch.min(data.imag)
        imag_data = (data.imag - m)/(M - m)
        
        return real_data + 1j * imag_data

    def get_data(self, n_samples):
        # given:    the total number of samples to get the data from
        # return:   the data in a tensor format

        input_data = np.zeros((n_samples, 1, self.in_size, self.in_size), dtype=np.cfloat)
        output_data = np.zeros((n_samples, 1, self.in_size, self.in_size), dtype=np.cfloat)

        for i in range(self.ntrain):
            # input data
            file_input  = self.file + "sample_" + str(i) + "_time_0.nc" 
            f = netCDF4.Dataset(file_input,'r')
            input_data[i, 0] = np.array(f.variables['u'][:] + 1j * f.variables['v'][:])
            f.close()

            # output data
            file_output = self.file + "sample_" + str(i) + "_time_1.nc" 
            f = netCDF4.Dataset(file_output,'r')
            output_data[i, 0] = np.array(f.variables['u'][:] + 1j * f.variables['v'][:])
            f.close()

        for i in range(self.ntest):
            # input data
            file_input  = self.file + "sample_" + str(896+i) + "_time_0.nc" 
            f = netCDF4.Dataset(file_input,'r')
            input_data[self.ntrain+i, 0] = np.array(f.variables['u'][:] + 1j * f.variables['v'][:])
            f.close()

            # output data
            file_output = self.file + "sample_" + str(896+i) + "_time_1.nc" 
            f = netCDF4.Dataset(file_output,'r')
            output_data[self.ntrain+i, 0] = np.array(f.variables['u'][:] + 1j * f.variables['v'][:])
            f.close()
            
        return torch.tensor(input_data).type(torch.cfloat), torch.tensor(output_data).type(torch.cfloat)
