import torch
import torch.utils.data as data
import os
import pickle
import numpy as np 
from PIL import Image
import os.path
import random

from torch.utils.data import Dataset
from tqdm import tqdm


def load_dataset(N=30000,NP=1800):

	obstacles=np.zeros((N,2800),dtype=np.float32)
	for i in range(0,N):
		temp=np.fromfile('../../data/S2D/dataset/obs_cloud/obc'+str(i)+'.dat')
		temp=temp.reshape(int(len(temp)/2),2)
		obstacles[i]=temp.flatten()

	return 	obstacles	


class AEDataset(Dataset):
	"""Auto-Encoder data loader
	"""
	def __init__(self, mode='train'):
		self.data = load_dataset()
		print(self.data.shape)
	
	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self, index):
		x = self.data[index]
		# x = torch.from_numpy(x).type(torch.float32)
		return x