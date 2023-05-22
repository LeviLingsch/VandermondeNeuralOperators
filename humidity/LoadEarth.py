import torch
import netCDF4 as nc
import numpy as np
import os
import pdb

class LoadEarth:
    def __init__(self, path):
        self.path = path
        self.file_names = os.listdir(path)
        self.names = ['CDH', 'CDQ', 'EFLUX','EVAP','FRCAN','FRCCN', 'FRCLS', 'HLML', 'QSTAR', 'QLML', 'SPEED', 'TAUX', 'TAUY', 'TLML', 'ULML', 'VLML']
        
        ds = nc.Dataset(f"{path}{self.file_names[0]}")
        self.lat, self.lon = np.meshgrid(ds['lon'][:], ds['lat'][:])
        self.y_shape = self.lat.shape[0]
        self.x_shape = self.lat.shape[1]
        
    def get_data(self, num_train, num_test, time_horizon=6):
        # given:    the number of training and testing samples
        # return:   the inputs and outputs of the selected data
        num_samples = (num_train + num_test)

        # inputs are several types of data all concatenated together, outputs are just QLML
        inputs = torch.zeros((num_samples, self.y_shape, self.x_shape, len(self.names)), dtype=torch.float)
        outputs = torch.zeros((num_samples, self.y_shape, self.x_shape, 1), dtype=torch.float)

        time_samples = 24 - time_horizon

        for sample in range(num_train//(24-time_horizon)):
            file = f"{self.path}{self.file_names[sample]}"
            data_set = nc.Dataset(file)

            for index, name in enumerate(self.names):
                inputs[sample*time_samples:(sample+1)*time_samples,:,:, index] = torch.tensor(data_set[name][:time_samples])

            outputs[sample*time_samples:(sample+1)*time_samples,:,:,0] = torch.tensor(data_set['QLML'][time_horizon:])

        for sample in range(num_test//(24-time_horizon)):
            file = f"{self.path}{self.file_names[-(sample+1)]}"
            data_set = nc.Dataset(file)

            for index, name in enumerate(self.names):
                inputs[num_train+sample*time_samples:num_train+(sample+1)*time_samples,:,:, index] = torch.tensor(data_set[name][:time_samples])

            outputs[num_train+sample*time_samples:num_train+(sample+1)*time_samples,:,:,0] = torch.tensor(data_set['QLML'][time_horizon:])

        inputs = self.normalize(inputs)
        outputs = self.normalize(outputs)

        train_in = inputs[:num_train]
        train_out = outputs[:num_train]

        test_in = inputs[-num_test:]
        test_out = outputs[-num_test:]
        
        return  train_in, train_out, test_in, test_out

    def normalize(self, data):
        normalized_data = torch.zeros_like(data)
        for index in range(data.shape[-1]):
            M = torch.max(data[:,:,:,index])
            m = torch.min(data[:,:,:,index])
            normalized_data[:,:,:,index] = (data[:,:,:,index] - m)/(M - m)
        return normalized_data


