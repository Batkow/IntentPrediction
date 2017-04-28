#!/usr/bin/python
import cv2

map = cv2.imread('gridworld/maps/cross_walk.png')
map = cv2.resize(map, (10,10))