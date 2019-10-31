import pytorch #??

TRAINING_STEPS_PER_ITERATION = 1000
HISTORY = 2

class NNET():

	def __init__(self):
		#initialise a nnet of known architecture and assign to self.net
		#in would be 13x13x(2xHISTORY + 1) and out would be 13X13 + 1

	def binarise_board(board):
		#given a board with 0, 1, -1 i want multiple boards which answer questions like where are black, where are whites
		#for go it would be just 2

	def preprocess(data):
		#list of  ([s0, s1, ...], pi, z)
		#gets converted to binary nd arrays as expected by nnet

	def train(self, data, steps = TRAINING_STEPS_PER_ITERATION):
		#pick minibatches of size 2048 from data and 

		#train for num of steps

	def copy(self):
		#create another instance of this object
		#deepcopy, there would be some pytorch command

		#return new instance
		
