import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import load_dataset 
from model import MLP 
from torch.autograd import Variable 
import math


import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def to_var(x, volatile=False):
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x, volatile=volatile)

def get_input(i,data,targets,bs):

	if i+bs<len(data):
		bi=data[i:i+bs]
		bt=targets[i:i+bs]	
	else:
		bi=data[i:]
		bt=targets[i:]
		
	return torch.from_numpy(bi),torch.from_numpy(bt)


    
def main(args):
	directory = './logs/' + datetime.datetime.now().strftime("%m%d_%H_%M/")
	writer = SummaryWriter(directory)
    
    
	# Build data loader
	dataset,targets= load_dataset(N=100, NP=4000, s=0, sp=0)
	print('training dataset loaded', len(dataset)) 
	test_dataset, test_target = load_dataset(N=100, NP=200, s=0, sp=4000)
	test_dataset_unseen, test_target_unseen = load_dataset(N=10, NP=2000, s=100, sp=0)
	print('dataset loaded') 
	
	# Build the models
	mlp = MLP(args.input_size, args.output_size)
    
	if torch.cuda.is_available():
		mlp.cuda()

	# Loss and Optimizer
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adagrad(mlp.parameters()) 
    
	# Train the Models
	sm=10 # start saving models after 100 epochs
	for epoch in tqdm(range(args.num_epochs), desc='epoch'):
		# print("epoch" + str(epoch))
		avg_loss=0
		for i in tqdm(range(0,len(dataset),args.batch_size), desc='batch', leave=False):
			# Forward, Backward and Optimize
			mlp.zero_grad()			
			bi,bt= get_input(i,dataset,targets,args.batch_size)
			bi=to_var(bi)
			bt=to_var(bt)
			bo = mlp(bi)
			loss = criterion(bo,bt)
			avg_loss=avg_loss+loss.item()
			loss.backward()
			optimizer.step()
		# print("--average loss:")
		# print(avg_loss/(len(dataset)/args.batch_size))
		writer.add_scalar('MPNETLoss/train_loss', avg_loss/(len(dataset)/args.batch_size), epoch)
		# Save the models
		if epoch==sm:
			model_path=directory + 'mpnet'+str(sm)+'.pkl'
			torch.save(mlp.state_dict(), model_path)
			sm=sm+50 # save model after every 50 epochs from 100 epoch ownwards
		
		if epoch % 10 == 0:
			test_on_val(test_dataset, test_target, args, mlp, writer, epoch, mode='')
			test_on_val(test_dataset_unseen, test_target_unseen, args, mlp, writer, epoch, mode='_unseen')

	model_path='mpnet_final.pkl'
	torch.save(mlp.state_dict(), model_path)


def test_on_val(test_dataset, test_target, args, mlp, writer, epoch, mode=''):
	criterion = nn.MSELoss()
	avg_loss=0
	with torch.no_grad():
		for i in range (0,len(test_dataset),args.batch_size):
			# Forward, Backward and Optimize		
			bi,bt= get_input(i,test_dataset,test_target,args.batch_size)
			bi=to_var(bi)
			bt=to_var(bt)
			bo = mlp(bi)
			loss = criterion(bo,bt)
			avg_loss=avg_loss+loss.item()
			writer.add_scalar('MPNETLoss/test_loss'+mode, avg_loss/(len(test_dataset)/args.batch_size), epoch)




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', type=str, default='./models/',help='path for saving trained models')
	parser.add_argument('--no_env', type=int, default=50,help='directory for obstacle images')
	parser.add_argument('--no_motion_paths', type=int,default=2000,help='number of optimal paths in each environment')
	parser.add_argument('--log_step', type=int , default=10,help='step size for prining log info')
	parser.add_argument('--save_step', type=int , default=1000,help='step size for saving trained models')

	# Model parameters
	parser.add_argument('--input_size', type=int , default=32, help='dimension of the input vector')
	parser.add_argument('--output_size', type=int , default=2, help='dimension of the input vector')
	parser.add_argument('--hidden_size', type=int , default=256, help='dimension of lstm hidden states')
	parser.add_argument('--num_layers', type=int , default=4, help='number of layers in lstm')

	parser.add_argument('--num_epochs', type=int, default=500)
	parser.add_argument('--batch_size', type=int, default=100)
	parser.add_argument('--learning_rate', type=float, default=0.0001)
	args = parser.parse_args()
	print(args)
	main(args)



