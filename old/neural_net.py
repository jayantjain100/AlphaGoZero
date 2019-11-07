import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import copy
TRAINING_STEPS_PER_ITERATION = 1000
HISTORY = 2
# BOARD_SIZE = 13
BOARD_SIZE = 5
# BOARD_SIZE = 3


# class NNET(nn.Module):
class NET(nn.Module):
# class NNET():

	def __init__(self):
		pass
		# super(Net, self).__init__()
		# self.conv1 = nn.Conv2d(2*HISTORY+1, 6, 3)
		# self.conv2 = nn.Conv2d(6, 16, 3)
		# self.fc1 = nn.Linear(16 * 6 * 6, 120)
		# self.fc2 = nn.Linear(120, 84)
  #       self.fc3 = nn.Linear(84, 10)
		#initialise a nnet of known architecture and assign to self.net
		#in would be 13x13x(2xHISTORY + 1) and out would be 13X13 + 1

	# def binarise_board(board):
	# 	#given a board with 0, 1, -1 i want multiple boards which answer questions like where are black, where are whites
	# 	#for go it would be just 2

	def stack(data, num):
		states, pi, z = data
		if (num % 2 == 0):
			C = np.ones((1,BOARD_SIZE,BOARD_SIZE))
		else:
			C = np.zeros((1,BOARD_SIZE,BOARD_SIZE))
		states.append(C)
		return (np.concatenate(states,axis = 0), pi, z)
	# def preprocess(data):
	# 	for da
	# 	#list of  ([s0, s1, ...], pi, z)
	# 	#gets converted to binary nd arrays as expected by nnet

	def train(self, data, steps = TRAINING_STEPS_PER_ITERATION):
		pass
		#pick minibatches of size 2048 from data and 

		#train for num of steps

	def copy(self):
		return copy.deepcopy(self)
		# pass
		#create another instance of this object
		#deepcopy, there would be some pytorch command

		#return new instance
		
	def forward_pass(self, inp):
		#inp is that bakwas 2*HISTORY + 1 jo chutiye ne diya hai
		l = [random.random() for i in range(BOARD_SIZE*BOARD_SIZE + 2)]
		l[BOARD_SIZE*BOARD_SIZE] = 0.001
		l[BOARD_SIZE*BOARD_SIZE + 1] = 0.001 #REMOVE LATER PENDING POTE
		# l[169] = 0
		# l[170] = 0 #PENDING, why is resign giving probs, POTE
		return l,random.random()
