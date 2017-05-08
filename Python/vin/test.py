#! /usr/bin/python

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from utils import *
import cv2
import math
import matplotlib.pyplot as plt
import random

# test functions


def dynamics(x,u,dt):
	xNext = np.array([x[0] + dt * np.cos(u),
        x[1] + dt * np.sin(u)])
	return xNext

def runAgent(map, value):

	nActions = 24
	nSteps = 2000
	actions = np.linspace(0,2*math.pi,nActions)
	x = np.array([[1.1], [1.1]])
	v = 0.1
	dt = 0.5
	X = x
	u = []
	goal = np.where(map[-1] == 1)
	goal = [goal[0][0], goal[1][0]]
	while(True):
	  xNew = dynamics(x,actions,dt)
	  vals = cv2.remap(value, xNew[0,:].astype('float32'), xNew[1,:].astype('float32'), cv2.INTER_LINEAR, borderValue = -1000)
	  maxVal = np.max(vals)
	  probs = np.exp(1000.0*(vals-maxVal))
	  totalProb = np.sum(probs)
	  cumProb = np.cumsum(probs/totalProb)

	  r = random.random()
	  idx = 0
	  while r > (cumProb[idx]):
	  	idx += 1

	  x = xNew[:,idx]

	  X = np.append(X,[[x[0]],[x[1]]],axis=1)
	  if (np.linalg.norm(x-goal) < 0.5): # 0.5 is good enough to see if it reaches goal
	  	break
	  u.append(actions[idx])

	return X



def checkValue(net, testloader, config, use_GPU):
    for i, data in enumerate(testloader): # Loop over batches of data
        X, labels = data
        X, labels = X.float(), labels.float()
        if use_GPU:
            X = X.cuda()
        X = Variable(X)
        outputs = net(X, config)
        prediction, label = outputs.cpu().squeeze().data.numpy(), labels.squeeze().numpy()
        out_img = np.concatenate((cv2.resize(prediction, (500,500)), cv2.resize(prediction, (500,500))), axis=1)
        if i == 10: # save an image to check
            out_img = cv2.normalize(out_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            cv2.imwrite("result_28x28.png",out_img)
    # cv2.imshow('prediction', cv2.resize(prediction,(500,500)))
    # cv2.imshow('label', cv2.resize(label,(500,500)))
    # cv2.waitKey(0)


def test(net, testloader, config, use_GPU):

    for i, data in enumerate(testloader): # Loop over batches of data
        X, labels = data
        X, labels = X.float(), labels.float()
        if use_GPU:
            X = X.cuda()
        inputMap = Variable(X)
        outputs = net(inputMap, config)
        prediction, label = outputs.cpu().squeeze().data.numpy(), labels.squeeze().numpy()
        print(label.max(), prediction.max())
        print(label.min(), prediction.min())
        semanticMap = X.squeeze().numpy()
        trajPredicted = runAgent(semanticMap, prediction)
        trajOptimal = runAgent(semanticMap, label)
        plt.imshow(prediction)
        plt.scatter(trajPredicted[0,:], trajPredicted[1,:],c='k',s=2)
        plt.scatter(trajOptimal[0,:], trajOptimal[1,:],c='m',s=2)
        plt.show()



