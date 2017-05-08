#!/usr/bin/python

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from utils import *

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
            # Loss (x 10 just to speed things up, could increase learning rate as well)
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










    
