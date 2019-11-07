import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

# TRAINING_STEPS_PER_ITERATION = 1000
TRAINING_STEPS_PER_ITERATION = 100
HISTORY = 2
# BOARD_SIZE = 13
BOARD_SIZE = 5
# NUMBER_OF_FILTERS = 32
NUMBER_OF_FILTERS = 8
# NUMBER_OF_RESIDUAL_BLOCKS = 9
NUMBER_OF_RESIDUAL_BLOCKS = 2
VALUE_HIDDEN = 256
# BATCH_SIZE = 2048
BATCH_SIZE = 32

class ResidualBlock(nn.Module):
	def __init__(self):
		super(ResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(NUMBER_OF_FILTERS, NUMBER_OF_FILTERS, stride = 1, kernel_size = 3, padding = 1)
		self.conv2 = nn.Conv2d(NUMBER_OF_FILTERS, NUMBER_OF_FILTERS, stride = 1, kernel_size = 3, padding = 1)
		self.bn1 = nn.BatchNorm2d(num_features = NUMBER_OF_FILTERS)
		self.bn2 = nn.BatchNorm2d(num_features = NUMBER_OF_FILTERS)

	def forward(self, x):
		hl1 = F.relu(self.bn1(self.conv1(x)))
		hl2 = self.bn2(self.conv2(hl1))
		skip = hl2 + x
		return F.relu(skip)

class PolicyHead(nn.Module):
	def __init__(self):
		super(PolicyHead, self).__init__()
		self.conv = nn.Conv2d(NUMBER_OF_FILTERS, 2, stride = 1, kernel_size = 1)
		self.bn = nn.BatchNorm2d(num_features = 2)
		self.fc = nn.Linear(2*(BOARD_SIZE - 2)*(BOARD_SIZE - 2), BOARD_SIZE*BOARD_SIZE+1)

	def forward(self, x):
		hl1 = F.relu(self.bn(self.conv(x)))
		hl1 = hl1.view(-1,2*(BOARD_SIZE - 2)*(BOARD_SIZE - 2))
		return self.fc(hl1)

class ValueHead(nn.Module):
	def __init__(self):
		super(ValueHead, self).__init__()
		self.conv = nn.Conv2d(NUMBER_OF_FILTERS, 1, stride = 1, kernel_size = 1)
		self.bn = nn.BatchNorm2d(num_features = 1)
		self.fc1 = nn.Linear((BOARD_SIZE - 2)*(BOARD_SIZE - 2), VALUE_HIDDEN)
		self.fc2 = nn.Linear(VALUE_HIDDEN, 1)

	def forward(self, x):
		hl1 = F.relu(self.bn(self.conv(x)))
		hl1 = hl1.view(-1, (BOARD_SIZE - 2)*(BOARD_SIZE - 2))
		hl2 = F.relu(self.fc1(hl1))
		return torch.tanh(self.fc2(hl2))

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv = nn.Conv2d(2*HISTORY+1, NUMBER_OF_FILTERS, stride = 1, kernel_size = 3)
		self.batch_norm = nn.BatchNorm2d(num_features = NUMBER_OF_FILTERS)
		self.res_blocks = []
		for i in range(NUMBER_OF_RESIDUAL_BLOCKS):
			self.res_blocks.append(ResidualBlock())
		self.policy_head = PolicyHead()
		self.value_head = ValueHead()

	def forward(self, x):
		hl1 = F.relu(self.batch_norm(self.conv(x)))
		for i in range(NUMBER_OF_RESIDUAL_BLOCKS):
			hl1 = self.res_blocks[i].forward(hl1)
		return self.policy_head.forward(hl1), self.value_head.forward(hl1)

	def predict(self, x):
		with torch.no_grad():
			x = torch.Tensor(np.expand_dims(x, axis = 0))
			hl1 = F.relu(self.batch_norm(self.conv(x)))
			for i in range(NUMBER_OF_RESIDUAL_BLOCKS):
				hl1 = self.res_blocks[i].forward(hl1)
			p,v = self.policy_head.forward(hl1), self.value_head.forward(hl1)
			p = p[0].detach().numpy()
			v = v.item()
			return p,v
class NNET():

	def __init__(self):
		pass
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

	def train(net, buff, steps = TRAINING_STEPS_PER_ITERATION):
		
		MSELoss = torch.nn.MSELoss()
		optimizer = torch.optim.SGD(net.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 10**-4)

		for epoch_num in range(num_epochs):
			optimizer.zero_grad()
			states, pis, zs = buff.sample(BATCH_SIZE)
			states, pis, zs = torch.Tensor(states), torch.Tensor(pis), torch.Tensor(zs)
			p, v = net(states)
			loss = - torch.sum(torch.mul(pis,p)) + MSELoss(zs,v)
			loss.backward()
			optimizer.step()
		#pick minibatches of size 2048 from data and 

		#train for num of steps

	# def copy(self):
	# 	pass
		#create another instance of this object
		#deepcopy, there would be some pytorch command

		#return new instance
		
	# def forward_pass(self, inp):
	# 	#inp is that bakwas 2*HISTORY + 1 jo chutiye ne diya hai
	# 	l = [random.random() for i in range(BOARD_SIZE*BOARD_SIZE + 2)]
	# 	# l[169] = 0
	# 	# l[170] = 0 #PENDING, why is resign giving probs, POTE
	# 	return l,random.random()

if __name__ == "__main__":
	nnet = NNET()
	inp = torch.Tensor(np.random.rand(10,5,13,13)) #batch size is 10
	nnet.forward(inp)
