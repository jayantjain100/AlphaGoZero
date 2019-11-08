#POTI means Potential Improvement
#POTE means Potential Error
#PENDING means knowingly pending choda hai (riya ko choda hai)
# ?? means idk kya likha hai, review laterx


#PENDING - feed KOMI value in environment by default
import numpy as np
from neural_net_2 import *
from go_game import *
import experience
import copy
import numpy as np
from global_constants import *

current_network = Net()
buff = experience.Memory()
iter_num = 0
while True:
	print('iter num is {}'.format(iter_num))
	iter_num += 1
	if iter_num % FREQUENCY_AGAINST_RANDOM == 0:
		print("Competing against random bot")
		res = compete_with_random(current_network, GAMES_AGAINST_RANDOM)
		print("percentage wins are {}%".format(100*res))

	#generate data from self play
	data = []
	print("self-play")
	for game in range(NUM_GAMES_PER_ITERATION):
		# print ("Game1")
		print('Games Played [%d%%]\r'%int((100*(game+1))/NUM_GAMES_PER_ITERATION), end="")
		game_data = play_single_for_training(current_network, show = SHOW)
		# print ("Game1 ended")
		#game_data is a list of (states, pi, z ) where a single state is a list of boards, list size = HISTORY
		#POTI - memory wastage
		game_data =  [NNET.stack(el,num) for num,el in enumerate(game_data)]

		# sys1.exit()
		data += game_data

	print ()
	print("self-play over")
	#data is a list of ([s0, s1, ..., s0+hiostory], pi, z)
	#convert the list of [s0, s1] to proper binary form as expected by the neural net
	#convert here because same data may be used by neural net multiplre times

	buff.add(data)

	# sys1.exit()
	#handles overflow

	#training
	# for _ in range(50):
	print("train")
	# print('Games Played [%d%%]\r'%int((100*game)/NUM_GAMES_PER_ITERATION), end="")
	# new_network = current_network.copy() #copy function for the NNET module
	if (PIT):
		new_network = copy.deepcopy(current_network) #copy function for the NNET module

		# new_network.train(buff)
		# train(buff)
		NNET.train(new_network, buff)
		# sys.exit()
		print ("pit")
		win_percentage = Environment.compete(new_network, current_network, verbose = SHOW)
		print ("The win percentage is {}".format(win_percentage))
		if win_percentage >= 0.55:
			current_network = new_network
	else:
		NNET.train(current_network, buff)



