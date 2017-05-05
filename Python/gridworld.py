#!/usr/bin/python

import numpy as np
import math
import cv2
from scipy.interpolate import interp2d
import random
import matplotlib.pyplot as plt
import copy
import sys

img = plt.imread('10x10grid.png')

def dynamics(x,u,dt):
	xNext = np.array([x[0] + dt * np.cos(u),
        x[1] + dt * np.sin(u)])
	return xNext


def valueIteration():

	global img

	gridHeight = 10
	gridWidth = 10
	nActions = 24
	discount = 0.99
	nIterations = 100
     
	actions = np.linspace(0,2*math.pi,nActions)

	rChannel = (img[:,:,0] == 1)
	gChannel = (img[:,:,1] == 1)
	bChannel = (img[:,:,2] == 1)

	R = np.zeros(img[:,:,0].shape)
	R[rChannel] = -2
	R[gChannel] = -0.5


	value = R.copy()

	while True:


		valueNew = value.copy()
		for xi in range(gridWidth):
			for yi in range(gridHeight):
				if ((xi == (gridWidth - 1)) and (yi == (gridWidth - 1))):
					continue
				Xnew = dynamics([xi, yi],actions,1)
				vals = cv2.remap(value, Xnew[0,:].astype('float32'), Xnew[1,:].astype('float32'), cv2.INTER_NEAREST, borderValue = -1000)
				#vals[vals == 0] = -1000 # temp fix for invalid locations
				total = R[yi,xi] + discount * np.max(vals)
				valueNew[yi,xi] = total

		normError = np.linalg.norm(value-valueNew)
		print normError

		if (normError < 0.0001):
			break

		value = valueNew.copy()



	valueImage = cv2.normalize(value, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	cv2.imshow('value', valueImage)
	cv2.waitKey(0)
	np.save('value.npy',value)
	return value


def runValueIterations(name,goalX,goalY):

	global img

	gridHeight = 10
	gridWidth = 10
	nActions = 24
	discount = 0.99
	nIterations = 100
     
	actions = np.linspace(0,2*math.pi,nActions)

	rChannel = (img[:,:,0] == 1)
	gChannel = (img[:,:,1] == 1)
	bChannel = (img[:,:,2] == 1)

	R = np.zeros(img[:,:,0].shape)
	R[rChannel] = -2
	R[gChannel] = -0.5
	R[goalY,goalX] = 0.0

	value = R.copy()

	while True:


		valueNew = value.copy()
		for xi in range(gridWidth):
			for yi in range(gridHeight):
				if ((xi == (goalX)) and (yi == (goalY))):
					continue
				Xnew = dynamics([xi, yi],actions,1)
				vals = cv2.remap(value, Xnew[0,:].astype('float32'), Xnew[1,:].astype('float32'), cv2.INTER_NEAREST, borderValue = -1000)
				#vals[vals == 0] = -1000 # temp fix for invalid locations
				total = R[yi,xi] + discount * np.max(vals)
				valueNew[yi,xi] = total

		normError = np.linalg.norm(value-valueNew)
		print normError

		if (normError < 0.0001):
			break

		value = valueNew.copy()



	valueImage = cv2.normalize(value, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	cv2.imshow('value', valueImage)
	cv2.waitKey(1)
	np.save('valueFolder/value' + name + '.npy',value)
	return value


def runAgent():

	global img
	value = np.load('valueFolder/value9x9.npy')
	#value = valueIteration(img)
	implot = plt.imshow(value)

	nActions = 24
	nSteps = 2000
	actions = np.linspace(0,2*math.pi,nActions)
	x = np.array([[1.1], [1.1]])
	v = 1
	dt = 1.0
	X = x
	u = []
	for k in range(nSteps):
	  xNew = dynamics(x,actions,dt)
	  vals = cv2.remap(value, xNew[0,:].astype('float32'), xNew[1,:].astype('float32'), cv2.INTER_NEAREST,borderValue = -1000)
	  #vals[vals == 0] = -1000
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
	  if (np.linalg.norm(x-np.array([[9],[9]])) < 1):
	  	break
	  u.append(actions[idx])

	plt.scatter(X[0,:], X[1,:],c='b',s=2)
	plt.show()



#valueIteration()
runAgent()

# for x in range(0,10):
# 	for y in range(0,10):
# 		name = str(x) + 'x' + str(y)
# 		runValueIterations(name,x,y)


# still not perfectly explored the map, but works. tweak later. add the negative reward for every time step.






