#!/usr/bin/python

import numpy as np
import math
import cv2
from scipy.interpolate import interp2d

def dynamics(x,u,dt):
	xNext = np.array([x[0] + dt * np.cos(u),
        x[1] + dt * np.sin(u)])
	return xNext


def main():

	gridHeight = 100
	gridWidth = 100
	nActions = 24
	discount = 0.999
	nIterations = 10
     
	actions = np.linspace(0,2*math.pi,nActions+1)
	np.delete(actions,-1)

	img = cv2.imread('intersection.png');
	rChannel = (img[:,:,2] == 255)
	gChannel = (img[:,:,1] == 255)
	bChannel = (img[:,:,0] == 255)

	R = np.zeros(img[:,:,0].shape)
	R[rChannel] = -2
	R[gChannel] = -0.3


	xStates = range(gridWidth)
	yStates = range(gridHeight)

	value = R
	interpFunction = interp2d(np.arange(value.shape[0]),np.arange(value.shape[1]),value)
	for k in range(nIterations):
		print(k)
		for xi in range(gridWidth):
			for yi in range(gridHeight):
				if ((xi == gridWidth-1) and (yi == gridWidth-1)):
					continue
				Xnew = dynamics([xi,yi],actions,1)
				total = R[yi,xi] + discount * np.max(interpFunction(Xnew[0,:],Xnew[1,:]))
				value[yi,xi] =  total

	
	#valueImage = cv2.normalize(value, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	#cv2.imshow('value', valueImage)
	#cv2.waitKey(0)

main()
