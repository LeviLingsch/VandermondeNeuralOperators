"""
@author: 
"""

"""
This code will train the VNO on the nonequispaced airfoil data, computing the Vandermonde matrices at run time.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer

from LoadCylinder import LoadCylinder
from VNO2D import VNO2D
from Transformation import VFT

import sys
sys.path.append('../')
from Adam import Adam
from utilities3 import *

import socket
import matplotlib.pyplot as plt

import pdb


def percentage_difference(truth, test):
    difference = torch.mean(torch.abs(truth.reshape(-1) - test.reshape(-1))/test.reshape(-1))
    return difference

def main():
    #########
    # configs
    #########
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    configs = {
        'num_train':            1500,
        'num_test':             300, 
        'batch_size':           20, 
        'epochs':               500,
        'modes':                12, 
        'width':                32,
        'learning_rate':        0.005,
        'scheduler_step':       10,
        'scheduler_gamma':      0.97,
        'display_predictions':  False,
        'save_model':           True,
        'min_max_norm':         False,
        'small_domain':         True,
    }
    file_path = print('Error: Modify this variable to be the path to the data!')
    print(configs)

    ##############
    # data loaders
    ##############
    loader = LoadCylinder(configs['num_train'], configs['num_test'], file_path=file_path)
    train_loader, test_loader = loader.get_dataloaders(configs['batch_size'], range_norm=configs['min_max_norm'], small_domain=configs['small_domain'])


    #######
    # model
    #######
    model_name = f"models/{configs['batch_size']}_{configs['modes']}_{configs['width']}_{configs['learning_rate']}"
    model = VNO2D(configs['modes'], configs['modes'], configs['width']).to(device)
    

    print(count_params(model))
    optimizer = Adam(model.parameters(), lr=configs['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configs['scheduler_step'], gamma=configs['scheduler_gamma'])
    myloss = torch.nn.L1Loss()
    
    ##########
    # training
    ##########
    for epoch in range(configs['epochs']):
        start_train = default_timer()
        train_loss = 0
        model.train()
        for iter, in_out in enumerate(train_loader):

            input = in_out[0].to(device)
            output = in_out[1].to(device)

            im = model(input)

            loss = myloss(output.reshape(-1), im.reshape(-1))
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        stop_train = default_timer()

        with torch.no_grad():
            test_loss = 0
            l1_relative_error = 0
            median_error = torch.zeros(configs['num_test'])
            iter=0
            model.eval()
            for in_out in test_loader:
                input = in_out[0].to(device)
                output = in_out[1].to(device)

                im = model(input)

                loss = myloss(output.reshape(-1), im.reshape(-1))
                test_loss += loss.item()


                l1_relative_error += percentage_difference(output, im)

                median_error[iter] = percentage_difference(output, im).item()
                iter+= 1

        scheduler.step()
        print(epoch, stop_train - start_train, train_loss / configs['num_train'], test_loss / configs['num_test'], torch.median(median_error).item())

    if configs['save_model']:
        print(f"model: VNO2D_{configs['epochs']}_{l1_relative_error/configs['num_test']:.4f}")
        torch.save(model, f"models/VNO2D_{configs['epochs']}_{l1_relative_error/configs['num_test']:.4f}")
    
    with torch.no_grad():
        test_loss = 0
        l1_relative_error = 0
        median_error = torch.zeros(configs['num_test'])
        iter=0
        model.eval()
        for in_out in test_loader:
            input = in_out[0].to(device)
            output = in_out[1].to(device)
            im = model(input)
            loss = myloss(output.reshape(-1), im.reshape(-1))
            test_loss += loss.item()
            l1_relative_error += percentage_difference(output, im)
            median_error[iter] = percentage_difference(output, im).item()
            print(median_error[iter].item())
            iter+= 1

        print(test_loss / configs['num_test'], torch.median(median_error).item())


if __name__=='__main__':
    main()