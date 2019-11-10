# mcst2.py
# directly copying the environment instead of recreating from start
import go_game
import copy
import numpy as np
from global_constants import *
import random
import sys
import os
import time
sys.path.append(os.path.abspath("../alphago_zero_sim"))
import goSim
# import ipdb
num_hits = 0

board_size = 13

class MonteCarloTreeNode():

	def sample(dist, temp):
		pass

	def __init__(self, parent, black, act, env, depth = 0):
		self.board = None
		self.parent = parent
		self.legal = None #POTI, remove faltu vars
		self.children = {} #action indexed
		self.leaf = True
		self.depth = depth
		# self.actions_till_now = a_till_now
		# if act is None:
		# 	self.actions_till_now = copy.deepcopy(a_till_now)
		# else:
		# 	self.actions_till_now = copy.deepcopy(a_till_now) + [act]
		self.prior = [] #P(s,a)
		self.visit_count = {} #N(s,a)
		self.w = {} #action indexed #W(s,a) 
		self.q = {} #Q(s,a)
		self.u = {} #PUCT algo vala

		self.black = black
		self.parent_env_ref = env
		self.terminal = False #how to handle this?? #POTE #PENDING
		self.a_to_reach_here = act

	def search(self):
		# print (self.depth)
		# if (self.depth > 500):
		# 	print ("FUCK YOU JAYANT")
		if self.leaf:
			return self
		else:
			node_count = sum(self.visit_count.values()) + 1
			rooted_count = node_count**0.5
			u = {a:((C_PUCT*self.prior[a]*rooted_count)/(1+self.visit_count[a])) for a in self.legal}
			f = {a:(u[a] + self.q[a]) for a in self.legal}
			chosen_action = max(f, key=f.get)

			child = self.children[chosen_action]

			return child.search()

	def fetch_history_and_append(self):
		current = self
		lis = []
		for i in range(HISTORY):
			lis.append(current.board)
			current = current.parent
			if current == None:
				break
		left_boards = HISTORY - len(lis)
		for i in range(left_boards):
			lis.append(np.zeros((2, BOARD_SIZE, BOARD_SIZE)))
		lis.reverse()
		return lis

	def expand(self, network):
		#run this on the found leaf
		if not self.leaf:
			raise Exception("NOT A LEAF")
		else:
			#keep on sampling
			self.leaf = False
			new_env = self.parent_env_ref.copy()
			if self.a_to_reach_here is not None:
				new_env.step(self.a_to_reach_here)
			self.env = new_env
			# self.board = env.restart_and_simulate_till(self.actions_till_now) 
			self.board = self.env.board #??, ask rajas
			if (self.env.ended):
				self.leaf = True
				# global num_hits
				# num_hits += 1
				# print('dummy escape')
				# sys.exit(0)
				return self.env.outcome

			if (self.depth >= 2*MOVE_CAP):
				self.leaf = True
				val = self.env.inner_env.state.board.official_score + self.env.inner_env.komi
				if (val < 0 and self.black) or (val > 0 and not self.black):
					return -1
				else:
					return 1

			self.legal = self.env.fetch_legal(self.black)
			# print (self.legal)
			if self.legal == []:
				raise Exception("Empty")

			# next_board = copy(env).step(a)[0] aisa kuch ayega next line mei
			# for a in self.legal:
			# 	new_env = self.env.copy()
			# 	new_env.step(a)
			# 	self.children[a] = MonteCarloTreeNode(self, not self.black, a , self.actions_till_now, env = new_env)	
			self.children = {a:MonteCarloTreeNode(self, not self.black, a, env =self.env, depth = self.depth + 1 ) for a in self.legal} 
			self.visit_count = {a:0 for a in self.legal} 
			self.w = {a:0 for a in self.legal}
			self.q = {a: (0) for a in self.legal} 

			for_forward_pass = self.fetch_history_and_append()
			if (self.black):
				for_forward_pass.append(np.ones((1,BOARD_SIZE,BOARD_SIZE)))
			else:
				for_forward_pass.append(np.zeros((1,BOARD_SIZE,BOARD_SIZE)))
			# self.prior , self.v = network.predict(np.concatenate(for_forward_pass, 0)) #PENDING
			self.prior = [1/len(self.legal) for i in range(BOARD_SIZE*BOARD_SIZE+1)]
			self.v = self.rollout_till_end(self.env)
			# print (self.prior.detach().numpy())
			#initial_val = ????
			# self.val_children = {a:initial_val for a in self.legal}


			# a = sample_from_children()

			return -self.v

	def propagate_back(self, v):
		current = self
		while (current.parent is not None):
			#go to parent
			a = current.a_to_reach_here
			par = current.parent
			par.visit_count[a] += 1
			par.w[a] += v
			par.q[a] = par.w[a]/(par.visit_count[a] + 1.)
			current = par
			v = -v

	def rollout_till_end(self, new_env):
		rollout_env = new_env.copy()
		done = False
		me = True
		while (not done):
			leg_coord = rollout_env.inner_env.state.board.get_legal_coords(rollout_env.current_color)
			a = goSim._coord_to_action(rollout_env.inner_env.state.board , random.choice(leg_coord))
			_, r, done = rollout_env.step(a)
			me = not me
		me = not me
		if (me):
			return r
		else:
			return -r

	def mcts(self, network, simulations = SIMULATIONS):
		# global num_hits
		# num_hits = 0
		start = time.time()
		for simnum in range(simulations):
			# print ("simulation number", simnum)
			chosen = self.search()
			v = chosen.expand(network)
			chosen.propagate_back(v)
		end = time.time()
		print (end - start)
		# print ("The number of hits is {}".format(num_hits))

		return self.visit_count






