#POTI means Potential Improvement
#POTE means Potential Error
#PENDING means knowingly pending choda hai
# ?? means idk kya likha hai, review laterx

import numpy as np
from neural_net import NNET
from go_game import environment
import experience
import copy
#module to train a new neural net
NUM_GAMES_PER_ITERATION = 25000


current_network = NNET()
buff = experience.Memory()

while True:
	#generate data from self play
	data = []
	for game in range(NUM_GAMES_PER_ITERATION):
		game_data = environment.play_single_for_training(current_network)
		#game_data is a list of (states, pi, z ) where a single state is a list of boards, list size = HISTORY
		#POTI - memory wastage
		data += game_data


	#data is a list of ([s0, s1, ...], pi, z)
	#convert the list of [s0, s1] to proper binary form as expected by the neural net
	#convert here because same data may be used by neural net multiplre times

	data = NNET.preprocess(data)

	buff.add(data)
	#handles overflow

	#training
	new_network = current_network.copy() #copy function for the NNET module
	new_network.train(buff)

	win_percentage = Environment.compete(new_network, current_network)

	if win_percentage >= 0.55:
		current_network = new_network



