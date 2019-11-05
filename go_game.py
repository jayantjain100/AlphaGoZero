#import lovishs game here??
import numpy as np
from neural_net import NNET
import mcts
import sys
import os
sys.path.append(os.path.abspath("../alphago_zero_sim"))
import copy

import goSim

BOARD_DIMS = (13, 13)
KOMI = 7.5
ACTIONS = BOARD_DIMS[0]*BOARD_DIMS[1] + 2 #pass, board0*board1 is pass, ans board0*board1 + 1 is resign
NUM_GAMES_COMPETITION = 5
HISTORY = 2

class Environment():
	def reset(self):
		self.inner_env.set_player_color(1)
		board = self.inner_env.reset()
		self.current_color = 1 # black
		self.board = board[:2]
		self.ended = False

	def __init__(self):
		self.inner_env = goSim.GoEnv(player_color = 'black', observation_type = 'image3c', illegal_move_mode = 'raise', board_size = BOARD_DIMS[0], komi = KOMI)
		self.current_color = 1
		board = self.inner_env.reset()
		self.board = board[:2]
		self.ended = False
		# self.board = np.zeros(BOARD_DIMS)#or lovishs func??
		#maybe just instantiate lovishs class here and set your own vars in accordance

	def step(self, a, show = False):
		#POTI - printing current score
		self.inner_env.set_player_color(self.current_color)
		self.current_color = 3 - self.current_color
		board, r, done, info, _ = self.inner_env.step(a)
		self.board = board[:2]
		self.ended = done
		self.outcome = r
		if show:
			print (info)
		return self.board, r, done
		# a is a single dimensional number 	break into dim1 and dim2 using BOARD_DIMS[0] and BOARD_DIMS[1]
		# returns board, 

	# def fetch_legal(self):
	# 	#returns a list of legal actions

	def fetch_legal(self, player):
		#POTE, PENDING history bhi chahiye hoga yahan??
		#returns a list of legal actions
		actions = []
		empties = np.ones(BOARD_DIMS) - (self.board[0] + self.board[1])
		tmp = np.stack([self.board[0],self.board[1], empties], 0)
		for i in range(ACTIONS):
			if (self.inner_env.is_legal_action(tmp, i, 1 if player else 2)):
				actions.append(i)
		return actions

	def restart_and_simulate_till(self, list_actions):
		self.reset()
		# print (list_actions, self.current_color)
		for a in list_actions:
			self.board, _, _ = self.step(a)

		return self.board

	def play_single_match(p1, p2):
		#returns 1True if p1 won 
		game = Environment()

		black = True
		board = np.zeros((2, BOARD_DIMS[0], BOARD_DIMS[1]))
		tree1 = mcts.MonteCarloTreeNode(None, black, None, [])
		tree2 = mcts.MonteCarloTreeNode(None, black, None, [])
		trees = {True: tree1, False:tree2}
		players = {True:p1, False: p2}
		done = False
		p1_wins = 0
		temperature = 0 #greedy play throughout 
		moves_till_now = []
		while(not done):
			tree = trees[black]
			network = players[black]
			visit_counts = tree.mcts(network, env = game) 
			game.restart_and_simulate_till(moves_till_now)

			a = normalise_and_sample(visit_counts, temperature)
			moves_till_now.append(a)
			_ , r , done = game.step(a)

			tree1 = tree1.children[a]
			tree2 = tree2.children[a]
			tree1.parent = None
			tree2.parent = None

			black = not black

		#ending mei if black is True then that means that black ki turn hai par game already over hai
		# so that means white won
		return not black

	def compete(p1, p2, num_games = NUM_GAMES_COMPETITION):
		#returns win percentage of p1
		#p1 wins
		# wins = {0:0, 1:0}
		# players = {0:p1, 1:p2}
		# for i in range(num_games):
		# 	if (play_single_for_training(players[i%2], players[(i+1)%2])):
		# 		wins[]
		#Assuming num_games is even

		p1_wins = 0
		for _ in range(int(num_games/2)):
			if (play_single_for_training(p1, p2)):
				#p1 won
				p1_wins += 1

		for _ in range(int(num_games/2)):
			if (not play_single_for_training(p2, p1)):
				p1_wins += 1

		return float(p1_wins) / float(num_games)


def sample(prob_dist):
	#prob_dist is a dictionary of keys with kiski kitni prob
	# returns action acc to prob dist
	net = 0.000000001
	
	random_float = np.random.rand()

	for k in prob_dist:
		if net > random_float:
			return k
		net += prob_dist[k]

	return k #POTE

	# raise Exception("NOT SAMPLED")

def normalise_and_sample(visit_counts, temperature = 1):
	s = sum(visit_counts.values())
	if temperature == 1:
		prob_dist =  {a:float(visit_counts[a])/s for a in visit_counts}
	else:
		chosen_action = max(visit_counts, key=visit_counts.get)
		prob_dist = {chosen_action : 1} #POTI , epsilon greedy

	a = sample(prob_dist)
	return a , prob_dist

def play_single_for_training(network):
	game = Environment()
	done = False
	data = []
	history = [np.zeros((2, BOARD_DIMS[0], BOARD_DIMS[1])) for _ in range(HISTORY)]
	#just keeping kaam ke 2 , where black, where white
	board = game.board
	black = True #player #POTE
	
	#initialise tree structure and pass with MCTS function
	tree = mcts.MonteCarloTreeNode(None, black, None, [])
	num_moves = 0
	temperature = 1
	moves_till_now = []
	while(not done):
		# print ("Num moves", num_moves)
		num_moves += 1
		if num_moves == 30: #could be changed acc to board size , POTI
			temperature = 0
		# print ("MCTS started")
		visit_counts = tree.mcts(network, game) #temperature??, #num_simulations
		# print ("MCTS ended")
		game.restart_and_simulate_till(moves_till_now)
		a, pi = normalise_and_sample(visit_counts, temperature) #pi is a dictionary
		# a = sample(visit_counts, temperature) #based on number of moves sets temp to 0 or 1 and chooses a move
		moves_till_now.append(a)
		true_pi = [0 for _ in range(ACTIONS)] #list
		for k in pi:
			true_pi[k] = pi[k]

		print ("Move no", len(moves_till_now))
		s_dash, r , done = game.step(a, True)
		# r , s_dash, done = game.step(a)
		
		#update history
		# history[:(HISTORY - 1)] = history[0:HISTORY]
		data.append((copy.deepcopy(history), true_pi) ) #check deepcopy, POTE
		# history[-1] = 
		history.pop(0)
		history.append(s_dash[:2]) #POTE

		black = not black
		tree = tree.children[a]
		tree.parent = None #POTE



	z = r
	l = len(data) - 1
	# if black:
	data = [(a,b, z*((-1)**(l-i))) for i,(a,b) in enumerate(data)] #POTE
	# else:
		# data = [(a,b, z*((-1)**(l-i+1))) for (a,b), i in enumerate(data)]

	return data