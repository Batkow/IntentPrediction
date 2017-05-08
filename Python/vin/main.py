#! /usr/bin/python

# test functions

import time
import argparse
#import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from dataset import *
from model import *
from utils import *
import cv2

from train import *
from test import *


if __name__ == '__main__':
    use_GPU = torch.cuda.is_available()
    # Parsing training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', 
                        type=str, 
                        default='training_data_28x28.npy', 
                        help='Path to data file')
    parser.add_argument('--imsize', 
                        type=int, 
                        default=28, 
                        help='Size of image')
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.005, 
                        help='Learning rate, [0.01, 0.005, 0.002, 0.001]')
    parser.add_argument('--trained', 
                        type=str, 
                        default='yes', 
                        help='use trained model, for test?')

    parser.add_argument('--epochs', 
                        type=int, 
                        default=500, 
                        help='Number of epochs to train')
    parser.add_argument('--k', 
                        type=int, 
                        default=50, 
                        help='Number of Value Iterations')
    parser.add_argument('--l_i', 
                        type=int, 
                        default=4, 
                        help='Number of channels in input layer')
    parser.add_argument('--l_h', 
                        type=int, 
                        default=150, 
                        help='Number of channels in first hidden layer')
    parser.add_argument('--l_q', 
                        type=int, 
                        default=24, 
                        help='Number of channels in q layer (~actions) in VI-module')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=100, 
                        help='Batch size')
    config = parser.parse_args()

    use_GPU = 0
    #Get path to save trained model
    save_path = "trained/vin_{0}x{0}.pth".format(config.imsize)

    if config.trained == 'yes':
    	net = torch.load(save_path)
    else:
    	# Instantiate a VIN model
    	net = VIN(config)
    print(net)
    trainset = GridworldData(config.datafile, imsize=config.imsize, train=True)
    testset = GridworldData(config.datafile, imsize=config.imsize, train=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=0) #use trainset shit, overfit
    criterion = nn.MSELoss()
    #optimizer = torch.optim.SGD(net.parameters(), lr = config.lr, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr = config.lr, betas=(0.9,0.999),eps=1e-08,weight_decay=0) #converges better
    train(net, trainloader, config, criterion, optimizer, use_GPU)
    checkValue(net, testloader, config, use_GPU)
    #test(net, testloader, config, use_GPU)
    torch.save(net, save_path)