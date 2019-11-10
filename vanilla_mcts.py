import go_game
import copy
import numpy as np
import random
import sys
import os
import time
sys.path.append(os.path.abspath("../alphago_zero_sim"))
import goSim
from global_constants import *

# C_PUCT = 1
# MOVE_CAP = 3*13*13
# SIMULATIONS = 100

class MonteCarloTreeNode():

	def __init__(self, parent, black, act, env = None, depth = 0):
		self.parent = parent
		self.children = {} #action indexed
		self.leaf = True
		self.depth = depth
		self.visit_count = {} #N(s,a)
		self.w = {} #action indexed #W(s,a)
		self.q = {} #Q(s,a)
		self.env = env

		self.black = black
		self.a_to_reach_here = act

	def search(self):
		if self.leaf:
			return self
		else:
			node_count = sum(self.visit_count.values()) + 1
			rooted_count = node_count**0.5
			uniform_prior = 1.0/len(self.legal)
			max_val = -1
			for a in self.legal:
				tmp = ((C_PUCT*(uniform_prior)*rooted_count)/(1+self.visit_count[a])) + self.q[a]
				if (tmp > max_val):
					max_val = tmp
					chosen_action = a
			child = self.children[chosen_action]
			return child.search()

	def expand(self):
		if not self.leaf:
			raise Exception("NOT A LEAF")
		else:
			self.leaf = False
			if (self.parent != None):
				self.env = self.parent.env.copy()

			if self.a_to_reach_here is not None:
				self.env.step(self.a_to_reach_here)

			if (self.env.ended):
				self.leaf = True
				return self.env.outcome

			if (self.depth >= 2*MOVE_CAP):
				self.leaf = True
				val = self.env.inner_env.state.board.official_score + self.env.inner_env.komi
				if (val < 0 and self.black) or (val > 0 and not self.black):
					return -1
				else:
					return 1

			self.legal = self.env.fetch_legal(self.black)
			if self.legal == []:
				raise Exception("Empty")

			self.children = {a:MonteCarloTreeNode(self, not self.black, a, self.depth + 1) for a in self.legal} 
			self.visit_count = {a:0 for a in self.legal}
			self.w = {a:0 for a in self.legal}
			self.q = {a:0 for a in self.legal}

			self.v = self.rollout_till_end()
			return -self.v

	def propagate_back(self, v):
		current = self
		while (current.parent != None):
			a = current.a_to_reach_here
			current.parent.visit_count[a] += 1
			current.parent.w[a] += v
			current.parent.q[a] = current.parent.w[a]/(current.parent.visit_count[a] + 1.)
			current = current.parent
			v = -v

	def rollout_till_end(self):
		rollout_env = self.env.copy()
		done = False
		me = True
		while (not done):
			leg_coord = rollout_env.inner_env.state.board.get_legal_coords(rollout_env.current_color)
			a = goSim._coord_to_action(rollout_env.inner_env.state.board, random.choice(leg_coord))
			_, r, done = rollout_env.step(a)
			me = not me
		me = not me
		if (me):
			return r
		else:
			return -r

	def mcts(self, network = None, simulations = SIMULATIONS):
		start = time.time()
		for simnum in range(simulations):
			chosen = self.search()
			v = chosen.expand()
			chosen.propagate_back(v)
		end = time.time()
		print (end - start)
		return self.visit_count
