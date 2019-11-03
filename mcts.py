import go_game
import copy
import numpy as np
# import ipdb
HISTORY = 2
#PENDING, POTI - rotations before passing to nnet, exploiting symmetry of the game
#PENDING - v_resign??

#same board could have multiple parents - think ply4 ?? permutations of same moves #POTI
BOARD_SIZE = 13
SIMULATIONS = 2
C_PUCT = 1 #PENDING ??

class MonteCarloTreeNode():

	def sample(dist, temp):
		pass

	def __init__(self, parent, black, act, a_till_now):
		self.board = None
		self.parent = parent
		self.legal = None #POTI, remove faltu vars
		self.children = {} #action indexed
		self.leaf = True
		if act is None:
			self.actions_till_now = copy.deepcopy(a_till_now)
		else:
			self.actions_till_now = copy.deepcopy(a_till_now + [act])
		self.prior = [] #P(s,a)
		self.visit_count = {} #N(s,a)
		self.w = {} #action indexed #W(s,a) 
		self.q = {} #Q(s,a)
		self.u = {} #PUCT algo vala

		self.black = black
		
		self.terminal = False #how to handle this?? #POTE #PENDING
		self.a_to_reach_here = act

	def search(self):
		if self.leaf:
			return self
		else:
			node_count = sum(self.visit_count.values())
			rooted_count = node_count**0.5
			u = {a:((C_PUCT*self.prior[a]*rooted_count)/(1+self.visit_count[a])) for a in self.legal}
			f = {a:(u[a] + self.q[a]) for a in self.legal}
			# print (self.legal)
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

	def expand(self, network, env):
		#run this on the found leaf
		if not self.leaf:
			raise Exception("NOT A LEAF")
		else:
			#keep on sampling
			self.leaf = False
			self.board = env.restart_and_simulate_till(self.actions_till_now) 
			if (env.ended):
				self.leaf = True
				# print('dummy escape')
				# sys.exit(0)
				return env.outcome
			self.legal = env.fetch_legal(self.black)
			if self.legal == []:
				raise Exception("Empty")

			# next_board = copy(env).step(a)[0] aisa kuch ayega next line mei
			self.children = {a:MonteCarloTreeNode(self, not self.black, a , self.actions_till_now) for a in self.legal}
			self.visit_count = {a:0 for a in self.legal} 
			self.w = {a:0 for a in self.legal}
			self.q = {a: (0) for a in self.legal} 

			for_forward_pass = self.fetch_history_and_append()
			if (self.black):
				for_forward_pass.append(np.ones((1,BOARD_SIZE,BOARD_SIZE)))
			else:
				for_forward_pass.append(np.zeros((1,BOARD_SIZE,BOARD_SIZE)))
			self.prior , self.v = network.forward_pass(np.concatenate(for_forward_pass, 0)) #PENDING
			
			#initial_val = ????
			# self.val_children = {a:initial_val for a in self.legal}


			# a = sample_from_children()

			return self.v

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

	def mcts(self, network, env, simulations = SIMULATIONS):
		for simnum in range(simulations):
			# print ("simulation number", simnum)
			chosen = self.search()
			v = chosen.expand(network, env)
			chosen.propagate_back(v)


		return self.visit_count




