"""
@author: 
"""

"""
This code will train the VNO
on the surface-level specific humidity data.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt


import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import socket
import sys
from MakeSparse2D import *
from LoadEarth import *
from Transformation import VFT2D
from VNO2D import VNO2DFixed
sys.path.append('../')
from Adam import Adam
from utilities3 import *
import pdb

torch.manual_seed(0)
np.random.seed(0)

################################################################
# configs
################################################################
configs = {
    'ntrain':           18*100,
    'ntest':            18*20,
    'modes':            32,
    'width':            32,
    'batch_size':       50,
    'epochs':           501,
    'learning_rate':    0.005,
    'scheduler_step':   10,
    'scheduler_gamma':  0.97,
    'center_lat':       180,
    'center_lon':       140,
    'uniform':          100,
    'growth':           2.0,
    'load_model':       False,
    'save_model':       True
}


def percentage_difference(truth, test):
    difference = torch.mean(torch.abs(truth - test)/truth)
    return difference.item()


################################################################
# main pipeline
################################################################
def main(configs):
    #########
    # configs
    #########
    print(configs)
    ntrain = configs['ntrain']  
    ntest = configs['ntest']  

    modes = configs['modes']  
    width = configs['width']  

    batch_size = configs['batch_size']  
    epochs = configs['epochs']  

    learning_rate = configs['learning_rate']  
    scheduler_step = configs['scheduler_step']  
    scheduler_gamma = configs['scheduler_gamma']  

    center_lat = configs['center_lat']
    center_lon = configs['center_lon']
    uniform = configs['uniform']  
    growth = configs['growth']  

    load_model = configs['load_model']  
    save_model = configs['save_model']  

    path_model =  print('Error: replace this variable with the path to the model.')
    file_path =  print('Error: replace this variable with the path to the data.')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ##############
    # data loaders
    ##############
    t1 = default_timer()
    print(f'Loading and processing data.')
    # load the data itself
    load_mod = LoadEarth(file_path)
    train_in, train_out, test_in, test_out = load_mod.get_data(ntrain, ntest)
    
    
    
    # create the needed distribution, x and y positions
    if growth == 1.0:
        y_pos = torch.tensor(load_mod.lon, dtype=torch.float)[:,0]
        x_pos = torch.tensor(load_mod.lat, dtype=torch.float)[0,:]
    else:
        sparsify = MakeSparse2D(train_in.shape[2], train_in.shape[1])
        x_pos, y_pos, nleft, nbelow = sparsify.generate_ce_distribution(growth, growth, center_lat, center_lon, uniform, uniform)
        train_in = sparsify.get_sparse_data(train_in, x_pos, y_pos)
        train_out = sparsify.get_sparse_data(train_out, x_pos, y_pos)
        test_in = sparsify.get_sparse_data(test_in, x_pos, y_pos)
        test_out = sparsify.get_sparse_data(test_out, x_pos, y_pos)

        l = nleft
        r = nleft+uniform
        b = nbelow
        t = nbelow+uniform


    t2 = default_timer()
    print(f'Processing finished in {t2-t1} seconds.')


    train_loader = DataLoader(TensorDataset(train_in, train_out), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_in, test_out), batch_size=1, shuffle=False)

    ##############
    # transformer
    ##############
    transformer = VFT2D(x_pos, y_pos, modes, modes, device)

    ##############
    # model
    ##############
    # initialize model
    if load_model:
        model = torch.load(path_model).to(device)
    else:
        model = VNO2DFixed(modes, modes, width, transformer, x_pos, y_pos).to(device)

    print(count_params(model))
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    myloss = torch.nn.L1Loss()

    ##########
    # training
    ##########
    for epoch in range(epochs):
        model.train()
        t1 = default_timer()

        train_l1_loss = 0
        for xx, yy in train_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            batch_size = xx.shape[0]
                
            im = model(xx)

            loss = myloss(im.reshape(batch_size, -1), yy.reshape(batch_size, -1))

            train_l1_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_l1_loss = 0
            l1_relative_error = 0
            for xx, yy in test_loader:
                loss = 0
                xx = xx.to(device)
                yy = yy.to(device)
                
                batch_size = xx.shape[0]

                im = model(xx)
                loss += myloss(im[:, b:t, l:r,:].reshape(batch_size, -1), yy[:, b:t, l:r,:].reshape(batch_size, -1))

                test_l1_loss += loss.item()

                l1_relative_error += percentage_difference(yy[:, b:t, l:r,:], im[:, b:t, l:r,:])
                
        t2 = default_timer()
        scheduler.step()
        print(epoch, t2 - t1, train_l1_loss / ntrain, test_l1_loss / ntest, l1_relative_error / ntest)

    if save_model:
        print(f"model: FNO2D_{epochs}_{test_l1_loss/ntest:.4f}")
        torch.save(model, f'./VNO2D_{epochs}_{test_l1_loss/ntest:.4f}')


    all_loss = torch.zeros(ntest)
    all_relative_error = torch.zeros(ntest)
    iter = 0
    for xx, yy in test_loader:
        xx = xx.to(device)
        yy = yy.to(device)

        batch_size = xx.shape[0]
            
        im = model(xx)

        loss = myloss(im[0, b:t, l:r,0].reshape(-1), yy[0, b:t, l:r,0].reshape(-1))

        test_l1_loss = loss.item()
        l1_relative_error = percentage_difference(yy[0, b:t, l:r,0], im[0, b:t, l:r,0])

        print(test_l1_loss, l1_relative_error)

        all_loss[iter]= test_l1_loss
        all_relative_error[iter] = l1_relative_error

        iter+=1
            
    print(torch.median(all_loss), torch.median(all_relative_error))
        

main(configs)
        
