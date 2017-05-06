#!/usr/bin/python

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






def train(net, trainloader, config, criterion, optimizer, use_GPU):
    for epoch in range(config.epochs): # Loop over dataset multiple times
        avg_error, avg_loss, num_batches = 0.0, 0.0, 0.0
        start_time = time.time()
        for i, data in enumerate(trainloader): # Loop over batches of data
            X, labels = data
            X, labels = X.float(), labels.float()
            if X.size()[0] != config.batch_size:
                continue
            if use_GPU:
                X = X.cuda()
                labels = labels.cuda() 
            X, labels = Variable(X), Variable(labels)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = net(X, config)
            # Loss
            loss = 10 * criterion(outputs, labels)
            # Backward pass
            loss.backward()
            # Update params
            optimizer.step()
            # Calculate Loss and Error
            loss_batch = loss.data[0]
            avg_loss += loss_batch
            num_batches += 1  
        time_duration = time.time() - start_time
        # Print epoch logs
        print_stats(epoch, avg_loss, num_batches, time_duration)
    print('\nFinished training. \n')


def test(net, testloader, config, use_GPU):
    for i, data in enumerate(testloader): # Loop over batches of data
        X, labels = data
        X, labels = X.float(), labels.float()
        if use_GPU:
            X = X.cuda()
        X = Variable(X)
        outputs = net(X, config)
        prediction, label = outputs.squeeze().data.numpy(), labels.squeeze().numpy()
        out_img = np.concatenate((cv2.resize(prediction, (500,500)), cv2.resize(prediction, (500,500))), axis=1)
        if i == 10: # save an image to check
            out_img = cv2.normalize(out_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            cv2.imwrite("result.png",out_img)
    # cv2.imshow('prediction', cv2.resize(prediction,(500,500)))
    # cv2.imshow('label', cv2.resize(label,(500,500)))
    # cv2.waitKey(0)




if __name__ == '__main__':
    use_GPU = torch.cuda.is_available()
    # Parsing training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', 
                        type=str, 
                        default='training_data.npy', 
                        help='Path to data file')
    parser.add_argument('--imsize', 
                        type=int, 
                        default=10, 
                        help='Size of image')
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.005, 
                        help='Learning rate, [0.01, 0.005, 0.002, 0.001]')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=500, 
                        help='Number of epochs to train')
    parser.add_argument('--k', 
                        type=int, 
                        default=20, 
                        help='Number of Value Iterations')
    parser.add_argument('--l_i', 
                        type=int, 
                        default=3, 
                        help='Number of channels in input layer')
    parser.add_argument('--l_h', 
                        type=int, 
                        default=150, 
                        help='Number of channels in first hidden layer')
    parser.add_argument('--l_q', 
                        type=int, 
                        default=8, 
                        help='Number of channels in q layer (~actions) in VI-module')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=100, 
                        help='Batch size')
    config = parser.parse_args()

    use_GPU = 0
    # Get path to save trained model
    #save_path = "trained/vin_{0}x{0}.pth".format(config.imsize) 
    # Instantiate a VIN model
    net = VIN(config)
    print(net)
    trainset = GridworldData(config.datafile, imsize=config.imsize, train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0) #use trainset shit, overfit
    criterion = nn.MSELoss()
    #optimizer = torch.optim.SGD(net.parameters(), lr = config.lr, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr = config.lr, betas=(0.9,0.999),eps=1e-08,weight_decay=0) #converges better
    train(net, trainloader, config, criterion, optimizer, use_GPU)
    test(net, testloader, config, use_GPU)


    
