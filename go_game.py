#import lovishs game here??
import numpy as np
from neural_net import NNET
import mcts

BOARD_DIMS = (13, 13)

ACTIONS = BOARD_DIMS[0]*BOARD_DIMS[1] + 1 #pass
NUM_GAMES_COMPETITION = 400

class Environment():

	def __init__(self):
		self.board = np.zeros(BOARD_DIMS)#or lovishs func??
		#maybe just instantiate lovishs class here and set your own vars in accordance

	def step(self, a):
		# a is a single dimensional number break into dim1 and dim2 using BOARD_DIMS[0] and BOARD_DIMS[1]
		# returns board, 

	# def fetch_legal(self):
	# 	#returns a list of legal actions

	def fetch_legal_2(board, player):
		#returns a list of legal actions

	def compete(p1, p2, num_games = NUM_GAMES_COMPETITION):
		#returns win percentage of p1


def sample(prob_dist):
	# returns action acc to prob dist
	pass


def play_single_for_training(network):
	game = Environment()
	done = False
	data = []
	history = [np.zeros(BOARD_DIMS) for _ in range(HISTORY)]

	board = game.board
	black = True #player #POTE
	
	#initialise tree structure and pass with MCTS function
	tree = mcts.MonteCarloTreeNode(None, board, black, None)

	while(not done):

		visit_counts = tree.mcts(network) #temperature??, #num_simulations
		a = sample(visit_counts, temperature) #based on number of moves sets temp to 0 or 1 and chooses a move


		r , s_dash, done = game.step(a)
		
		#update history
		# history[:(HISTORY - 1)] = history[0:HISTORY]
		# history[-1] = 
		history.pop(0)
		history.append(s_dash) #POTE

		data.append((copy.deepcopy(history), pi) ) #check deepcopy, POTE
		black = not black
		tree = tree.children[a]
		tree.parent = None #POTE



	z = r
	l = len(data) - 1
	# if black:
	data = [(a,b, z*((-1)**(l-i))) for (a,b), i in enumerate(data)] #POTE
	# else:
		# data = [(a,b, z*((-1)**(l-i+1))) for (a,b), i in enumerate(data)]

	return data