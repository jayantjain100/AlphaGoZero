#implement a cyclic buffer
import random
import numpy as np
from global_constants import *

class Memory():
	def __init__(self, size = BUFFER_SIZE):
		self.store = [None for _ in range(size)]
		self.full = 0
		self.cap = size
		self.start = 0

	def add(self, data):
		#data is a list
		if self.start + len(data) >= self.cap:
			#break data in 2 parts
			part1 = self.cap - self.start
			part2 = len(data) - part1
			self.store[self.start:] = data[:part1]
			self.store[:part2] = data[part1:]
			self.start = part2
		else:
			new_start = self.start + len(data)
			self.store[self.start:new_start] = data
			self.start = new_start

		self.full = min(self.full + len(data), self.cap)

	def sample(self, num_samples):
		indices = random.sample(list(range(self.full)), num_samples)
		states = np.array([self.store[i][0] for i in indices])
		pis = np.array([self.store[i][1][:-1] for i in indices])
		zs = np.array([self.store[i][2] for i in indices])
		zs = zs.reshape(-1,1)
		return states, pis, zs


