import numpy as np

import torch
import torch.utils.data as data


class GridworldData(data.Dataset):
    def __init__(self, file, imsize, train=True):
        assert file.endswith('.npy') # Must be .npy format
        self.file = file
        self.imsize = imsize
        self.train = train
        self.inputs, self.labels = self._process(file, self.train)


    def __getitem__(self, index):
    	img = self.inputs[index]
    	label = self.labels[index]
    	return img, label

    def __len__(self):
    	return self.inputs.shape[0]


    def _process(self, file, train):
   	  dataset = np.load(file)
   	  inputs = dataset[:,0:4,:,:]
   	  labels = dataset[:,4,:,:]
   	  return inputs, labels
