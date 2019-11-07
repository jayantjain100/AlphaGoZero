import go_game
import copy
#PENDING, POTI - rotations before passing to nnet, exploiting symmetry of the game
#PENDING - v_resign??

#same board could have multiple parents - think ply4 ?? permutations of same moves #POTI

SIMULATIONS = 1600
C_PUCT = 1 #PENDING ??

class MonteCarloTreeNode():

	def sample(dist, temp):
		pass

	def __init__(self, parent, board, black, act = -1):
		self.board = board
		self.parent = parent
		self.legal = [] #POTI, remove faltu vars
		self.children = {} #action indexed
		self.leaf = True

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
			# child = sample_from_children()#PENDING 
			node_count = sum(self.visit_count.values())
			rooted_count = node_count**0.5
			u = {a:((C_PUCT*self.prior[a]*rooted_count)/(1+self.visit_count[a])) for a in self.legal}
			f = {a:(u[a] + self.q[a]) for a in self.legal}
			chosen_action = max(f, key=d.get)

			child = self.children[chosen_action]

			return child.search()

	def fetch_history_and_append(self):
		current = self
		lis = []
		#POTE - fix this, this is not till the top, it is only till the HISTORYs
		while(current is not None ):
			lis.append(current.board)

		return lis.reverse()

	def expand(self, network):
		#run this on the found leaf
		if not self.leaf:
			raise Exception("NOT A LEAF")
		else:
			#keep on sampling
			self.leaf = False
			self.legal = go_game.fetch_legal_2(self.board, self.black)
			if self.legal == []:
				raise Exception("Empty")

			# next_board = copy(env).step(a)[0] aisa kuch ayega next line mei
			self.children = {a:MonteCarloTreeNode(self, next_board, not self.black, a ) for a in self.legal} #PENDING
			self.visit_count = {a:0 for a in self.legal} 
			self.w = {a:0 for a in self.legal}
			self.q = {a: (0) for a in self.legal} 

			self.prior , self.v = network.forward_pass(self.fetch_history_and_append()) #PENDING
			
			#initial_val = ????
			# self.val_children = {a:initial_val for a in self.legal}


			# a = sample_from_children()

			return self.v

	def propagate_back(self, v):
		current = self
		while(current is not None):
			#go to parent
			a = current.a_to_reach_here
			par = current.parent
			par.visit_count[a] += 1
			par.w[a] += v
			par.q[a] = par.w[a]/(par.visit_count[a] + 1.)
			# par.val_children = update here #PENDING
			current = par

	def mcts(self, network, simulations = SIMULATIONS):
		for _ in range(simulations):
			chosen = self.search()
			v = chosen.expand(network)
			chosen.propagate_back(v)


		return self.visit_count




