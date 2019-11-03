#implement a cyclic buffer
BUFF_GAMES = 50
AVG_GAME_LENGTH = 150
BUFFER_SIZE = BUFF_GAMES * AVG_GAME_LENGTH

class Memory():
	def __init__(self, size = BUFFER_SIZE):
		self.store = [None for _ in range(size)]
		self.full = 0
		self.cap = size
		self.start = 0

	def add(self, data):
		#data is a list
		if self.start + len(data) >= self.cap:
			#break data in 2 parts
			part1 = self.cap - self.start
			part2 = len(data) - part1
			self.store[self.start:] = data[:part1]
			self.store[:part2] = data[part1:]
			self.start = part2
		else:
			new_start = self.start + len(data)
			self.store[self.start:new_start] = data
			self.start = new_start

		self.full = min(self.full + len(data), self.cap)

	def sample(self, num_samples):
		#PENDING
		pass


