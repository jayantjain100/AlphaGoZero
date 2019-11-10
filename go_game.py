#import lovishs game here??
import numpy as np
from neural_net_2 import *
# import mcts as mcts
import vanilla_mcts as mcts
import sys
import os
sys.path.append(os.path.abspath("../alphago_zero_sim"))
import copy
import goSim
import random
from global_constants import *
import time

class Environment():
	def reset(self):
		self.inner_env.set_player_color(1)
		board = self.inner_env.reset()
		self.current_color = 1 # black
		self.board = board[:2]
		self.ended = False
		self.outcome = None
		# self.num_moves = 0

	def __init__(self):
		self.inner_env = goSim.GoEnv(player_color = 'black', observation_type = 'image3c', illegal_move_mode = 'raise', board_size = BOARD_DIMS[0], komi = KOMI)
		self.current_color = 1
		board = self.inner_env.reset()
		self.board = board[:2]
		self.ended = False
		self.outcome = None
		# self.num_moves = 0
		# self.board = np.zeros(BOARD_DIMS)#or lovishs func??
		#maybe just instantiate lovishs class here and set your own vars in accordance

	def step(self, a, show = False):
		#POTI - printing current score
		self.inner_env.set_player_color(self.current_color)
		self.current_color = 3 - self.current_color
		board, _, r, done, info, _ = self.inner_env.step(a)
		self.board = board[:2]
		self.ended = done
		self.outcome = r
		# self.num_moves += 1

		# if self.num_moves == MOVE_CAP:
		# 	return self.board, 0, True
		if show:
			if a == BOARD_SIZE * BOARD_SIZE:
				print("player passed")
			if a == BOARD_SIZE *BOARD_SIZE + 1:
				print("player resigned")
			print (info)
			if done:
				print()
				# print("{} the game_________________________ ".format(("WON" if r==1 else "LOST")))
				val = self.inner_env.state.board.official_score + self.inner_env.komi
				print("{} won the game and val was {}".format(("BLACK" if val < 0 else "WHITE"), val))
				print()
		return self.board, r, done
		# a is a single dimensional number 	break into dim1 and dim2 using BOARD_DIMS[0] and BOARD_DIMS[1]
		# returns board, 

	# def fetch_legal(self):
	# 	#returns a list of legal actions
	def copy(self):
		new_env = copy.copy(self)
		new_env.inner_env = copy.copy(self.inner_env)
		new_env.ended = self.ended
		new_env.current_color = self.current_color
		new_env.board = copy.deepcopy(self.board)
		new_env.outcome = self.outcome
		return new_env


	def fetch_legal(self, player):
		l = self.inner_env.state.board.get_legal_coords(1 if player else 2)
		l = [goSim._coord_to_action(self.inner_env.state.board , i) for i in l]
		return l

	def play_single_match_2(p1, p2, show = False):
		#returns 1True if p1 won 
		game = Environment()

		black = True
		done = False
		num_moves = 0
		while(not done):
			num_moves += 1

			tree = mcts.MonteCarloTreeNode(None, black, None, env = game)
			if black:
				visit_counts = tree.mcts(p1)
			else:
				visit_counts = tree.mcts(p2)

			a, _ = normalise_and_sample(visit_counts, 0)
			_ , r , done = game.step(a, show)

			if num_moves == MOVE_CAP:
				done = True
				#decide winner
				val = game.inner_env.state.board.official_score + game.inner_env.komi
				if (val > 0):
					winner = -1
					return winner
				else:
					winner = 1
					return winner

			black = not black

		black = not black
		if (black):
			winner = r
		else:
			winner = -r	

		return winner

	def play_single_match_with_random(p1,my_black, verbose = False):
		game = Environment()

		black = True
		done = False
		num_moves = 0
		while(not done):
			num_moves += 1
			if (black == my_black):
				tree = mcts.MonteCarloTreeNode(None, black, None, env = game)
				visit_counts = tree.mcts(p1)
				a, pi = normalise_and_sample(visit_counts, 0)
				# print(visit_counts, pi, tree.prior)
				_ , r , done = game.step(a, verbose)
				if num_moves == MOVE_CAP:
					done = True
					val = game.inner_env.state.board.official_score + game.inner_env.komi
					if (val > 0):
						winner = -1
						return winner
					else:
						winner = 1
						return winner
			else:
				a = random.choice(game.fetch_legal(black))
				_ , r , done = game.step(a, verbose)
				if num_moves == MOVE_CAP:
					done = True
					val = game.inner_env.state.board.official_score + game.inner_env.komi
					if (val > 0):
						winner = -1
						return winner
					else:
						winner = 1
						return winner
			black = not black

		black = not black
		if (black):
			winner = r
		else:
			winner = -r	

		return winner

	def compete(p1, p2, num_games = NUM_GAMES_COMPETITION, verbose = False):
		#returns win percentage of p1
		#p1 wins
		# wins = {0:0, 1:0}
		# players = {0:p1, 1:p2}
		# for i in range(num_games):
		# 	if (play_single_for_training(players[i%2], players[(i+1)%2])):
		# 		wins[]
		#Assuming num_games is even

		p1_wins = 0
		p2_wins = 0
		game = 0
		for _ in range(int(num_games/2)):
			game += 1
			ret = Environment.play_single_match_2(p1, p2)
			if (ret == 1):
				p1_wins += 1
			elif ret == -1:
				p2_wins += 1
			if verbose:
				print('Games Played [%d%%]\r'%int((100*game)/NUM_GAMES_COMPETITION), end="")
		for _ in range(int(num_games/2)):
			game += 1
			ret = Environment.play_single_match_2(p2, p1)
			if (ret == 1):
				p2_wins += 1
			elif ret == -1:
				p1_wins += 1
			if verbose:
				print('Games Played [%d%%]\r'%int((100*game)/NUM_GAMES_COMPETITION), end="")
		print()
		draws = (num_games - p1_wins - p2_wins)
		print ("The number of draws is {} out of {}".format(draws,num_games))
		return float(p1_wins + (draws*0.5)) / float(num_games)

def compete_with_random(p1, num_games = GAMES_AGAINST_RANDOM, verbose = False):
	my_wins = 0
	draws = 0
	im_black = True
	for i in range(num_games):
		
		# print('Games Played against random [%d%%]\r'%int((100*(i))/num_games), end="")

		ret = Environment.play_single_match_with_random(p1, im_black, verbose)
		if ((im_black and ret == 1) or (not im_black and ret == -1)):
			my_wins += 1
		elif (ret == 0):
			draws += 1
		im_black = not im_black
	# print ("The number of draws is {} out of {}".format(draws,num_games))		
	return float(my_wins + (draws*0.5)) / float(num_games)

def sample(prob_dist):
	#prob_dist is a dictionary of keys with kiski kitni prob
	# returns action acc to prob dist
	# net = 0.000000001
	net = 0
	
	random_float = np.random.rand()

	for k in prob_dist:
		net += prob_dist[k]
		if net >= random_float:
			return k

	raise Exception("Sampling Error")
	# return k #POTE

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

def play_single_for_training(network, show = False):
	game = Environment()
	done = False
	data = []
	history = [np.zeros((2, BOARD_DIMS[0], BOARD_DIMS[1])) for _ in range(HISTORY)]
	#just keeping kaam ke 2 , where black, where white
	board = game.board
	black = True #player #POTE
	
	#initialise tree structure and pass with MCTS function
	tree = mcts.MonteCarloTreeNode(None, black, None ,env = game)
	num_moves = 0
	temperature = 1
	moves_till_now = []
	while(not done):
		# print ("Num moves", num_moves)
		num_moves += 1
		
		if num_moves == TEMP_THRESHOLD_MOVES: #could be changed acc to board size , POTI
			temperature = 0
		# print ("MCTS started")
		visit_counts = tree.mcts(network) #temperature??, #num_simulations
		# print ("MCTS ended")
		# game.restart_and_simulate_till(moves_till_now)
		a, pi = normalise_and_sample(visit_counts, temperature) #pi is a dictionary
		# a = sample(visit_counts, temperature) #based on number of moves sets temp to 0 or 1 and chooses a move
		moves_till_now.append(a)
		# print(black, pi, game.fetch_legal(black))
		# print (tree.legal)
		true_pi = [0 for _ in range(ACTIONS)] #list
		for k in pi:
			true_pi[k] = pi[k]

		# print ("Move no", len(moves_till_now))
		s_dash, r , done = game.step(a, show)
		# r , s_dash, done = game.step(a)
		if num_moves == MOVE_CAP :
			done = True
			val = game.inner_env.state.board.official_score + game.inner_env.komi
			if (val < 0 and black) or (val > 0 and not black):
				r = 1
			else:
				r = -1
		#update history
		# history[:(HISTORY - 1)] = history[0:HISTORY]
		data.append((copy.deepcopy(history), true_pi) ) #check deepcopy, POTE
		# history[-1] = 

		history.pop(0)
		history.append(s_dash[:2]) #POTE

		black = not black
		tree = tree.children[a]
		tree.parent = None #POTE

	# sys1.exit()

	z = r
	l = len(data) - 1
	# if black:
	data = [(a,b, z*((-1)**(l-i))) for i,(a,b) in enumerate(data)] #POTE
	# else:
		# data = [(a,b, z*((-1)**(l-i+1))) for (a,b), i in enumerate(data)]

	return data