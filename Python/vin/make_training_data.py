#!/usr/bin/python

import numpy as np
import math
import cv2
from scipy.interpolate import interp2d
import random
import matplotlib.pyplot as plt
import copy
import sys
import pickle

#img = plt.imread('10x10grid.png')
img = cv2.resize(plt.imread('../maps/cross_walk_multiple.png'),(28,28))

def dynamics(x,u,dt):
	xNext = np.array([x[0] + dt * np.cos(u),
        x[1] + dt * np.sin(u)])
	return xNext



def runValueIterations(name, gridSize, goalX, goalY):

	global img

	gridHeight = gridSize[0]
	gridWidth = gridSize[1]
	nActions = 24
	discount = 0.99
     
	actions = np.linspace(0,2*math.pi,nActions)

	rChannel = (img[:,:,0] == 1)
	gChannel = (img[:,:,1] == 1)
	bChannel = (img[:,:,2] == 1)

	R = np.zeros(img[:,:,0].shape)
	R[rChannel] = -2
	R[gChannel] = -0.5
	R[bChannel] = -1 # will only be used if there are 3 features
	R[goalY,goalX] = 1.0

	value = R.copy()

	while True:
		valueNew = value.copy()
		for xi in range(gridWidth):
			for yi in range(gridHeight):
				if ((xi == (goalX)) and (yi == (goalY))):
					continue
				Xnew = dynamics([xi, yi],actions,1)
				vals = cv2.remap(value, Xnew[0,:].astype('float32'), Xnew[1,:].astype('float32'), cv2.INTER_LINEAR, borderValue = -1000)
				#vals[vals == 0] = -1000 # temp fix for invalid locations
				total = R[yi,xi] + discount * np.max(vals)
				valueNew[yi,xi] = total

		normError = np.linalg.norm(value-valueNew)
		#print normError

		if (normError < 0.1): # 0.1 is good enough
			break

		value = valueNew.copy()

	inp = np.zeros((4,gridHeight, gridWidth))
	inp[0] = rChannel.astype(int)
	inp[1] = gChannel.astype(int)
	inp[2] = bChannel.astype(int)
	inp[3,goalY,goalX] = int(1)

	label = value

	minValue, maxValue = np.min(label), np.max(label)
	label = (label - minValue)/(maxValue - minValue)

	valueImage = cv2.resize(cv2.normalize(label, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F),(100,100))
	cv2.imshow('value', cv2.resize(valueImage,(500,500)))
	cv2.waitKey(1)

	return inp, label



def makeTrainingData(gridSize = (10,10)):

	dataset = np.zeros((gridSize[0] * gridSize[1],5,gridSize[0],gridSize[1]))
	for x in range(0,gridSize[0]):
		for y in range(0,gridSize[1]):
			name = str(x) + 'x' + str(y)
			inputD, labelD = runValueIterations(name, gridSize, x, y)
			#print(inputD.shape, labelD.shape)
			dataset[x*gridSize[0] + y,0:4,:,:] = inputD
			dataset[x*gridSize[0] + y, 4,:,:] = labelD
		print(x)
	np.save('training_data_28x28.npy', dataset)
	print('Done saving training data')


makeTrainingData(gridSize = (28,28))
