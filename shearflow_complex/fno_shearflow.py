"""
@author: 
"""

"""
This code will train the FNO (Li et al., Fourier Neural Operator for Parametric Partial Differential Equations, https://arxiv.org/abs/2010.08895)
on the shear layer data.
"""


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import socket
import sys
from MakeSparse2D import *
from LoadShearflow import *
from FNO2D import FNO2D
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
    'ntrain':           896,
    'ntest':            128,
    'modes':            20,
    'width':            32,
    'batch_size':       2,
    'epochs':           101,
    'learning_rate':    0.005,
    'scheduler_step':   10,
    'scheduler_gamma':  0.97,
    'center_1':         256,
    'center_2':         768,
    'uniform':          100,
    'growth':           1.75,
    'load_model':       False,
    'save_model':       True,
    'select_sparse':    True
}


def percentage_difference(truth, test):
    difference = torch.mean(torch.abs(truth - test)/torch.mean(truth))
    # difference = torch.mean(torch.abs(truth - test)/truth)
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

    center_points = [configs['center_1']  , configs['center_2']  ]
    uniform = configs['uniform']  
    growth = configs['growth']  

    load_model = configs['load_model']  
    save_model = configs['save_model']   

    select_sparse = configs['select_sparse']

    if socket.gethostname() == 'SRL-DSK-004':
        file_path = '../'
    else:
        file_path='/cluster/scratch/llingsch/ShearFlow/ddsl/'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ##############
    # data loaders
    ##############
    t1 = default_timer()
    print(f'Loading and processing data.')
    # load the data itself
    load_mod = LoadShearflow(ntrain, ntest, file=file_path)
    train_a, train_u, test_a, test_u = load_mod.return_data()
    
    # create the needed distribution, x and y positions
    sparsify = MakeSparse2D(train_a.shape[2], train_a.shape[1])
    sparse_x = sparsify.shear_distribution(train_a, center_points, growth, uniform)[1]
    sparse_x = sparse_x.int()
    y_pos = torch.arange(1024)

    # create data loaders
    train_loader = DataLoader(TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
    t2 = default_timer()
    print(f'Processing finished in {t2-t1} seconds.')


    ##############
    # model
    ##############
    # initialize model
    if load_model:
        model = torch.load(path_model).to(device)
    else:
        model = FNO2D(modes, modes, width).to(device)
        path_model = f'/cluster/scratch/llingsch/ShearFlow/models/'

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

            real_loss = myloss(im.real.reshape(batch_size, -1), yy.real.reshape(batch_size, -1))
            imag_loss = myloss(im.imag.reshape(batch_size, -1), yy.imag.reshape(batch_size, -1))
            loss = real_loss + imag_loss

            train_l1_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_l1_loss = 0
            real_relative_error = 0
            imag_relative_error = 0
            for xx, yy in test_loader:
                loss = 0
                xx = xx.to(device)
                yy = yy.to(device)
                
                batch_size = xx.shape[0]

                im = model(xx)
                if select_sparse:
                    im = torch.index_select(im, 2, sparse_x.to(device))
                    yy = torch.index_select(yy, 2, sparse_x.to(device))

                real_loss = myloss(im.real.reshape(batch_size, -1), yy.real.reshape(batch_size, -1))
                imag_loss = myloss(im.imag.reshape(batch_size, -1), yy.imag.reshape(batch_size, -1))
                loss = real_loss + imag_loss

                test_l1_loss += loss.item()

                real_relative_error += percentage_difference(yy.real, im.real)
                imag_relative_error += percentage_difference(yy.imag, im.imag)
                
        t2 = default_timer()
        scheduler.step()
        print(epoch, t2 - t1, train_l1_loss / ntrain, test_l1_loss / ntest, real_relative_error / ntest, imag_relative_error / ntest)

    if save_model and not socket.gethostname() == 'SRL-DSK-004':
        print(f"model: FNO2D_{epochs}_{test_l1_loss:.4f}")
        torch.save(model, f'/cluster/scratch/llingsch/ShearFlow/models/FNO2D_complex_{epochs}_{real_relative_error / ntest:.4f}')
    
    # compute median error
    model.eval()
    with torch.no_grad():
        all_loss = torch.zeros(ntest)
        all_relative_error_real = torch.zeros(ntest)
        all_relative_error_imag = torch.zeros(ntest)
        iter = 0
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            
            batch_size = xx.shape[0]
            im = model(xx)
            loss = myloss(im.real.reshape(batch_size, -1), yy.real.reshape(batch_size, -1))

            test_l1_loss = loss.item()
            l1_relative_error_real = percentage_difference(yy.real, im.real)
            l1_relative_error_imag = percentage_difference(yy.imag, im.imag)

            print(test_l1_loss, l1_relative_error_real, l1_relative_error_imag)
            all_loss[iter]= test_l1_loss
            all_relative_error_real[iter] = l1_relative_error_real
            all_relative_error_imag[iter] = l1_relative_error_imag
            iter+=1
        
        print(torch.median(all_loss), torch.median(all_relative_error_real), torch.median(all_relative_error_imag))

main(configs)