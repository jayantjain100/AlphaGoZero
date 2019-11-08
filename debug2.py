import torch
from go_game import *
from neural_net_2 import *
current_network = torch.load("5_cross_5_iter_40.pt")
# current_network = Net()
print("no of simulations are {}".format(SIMULATIONS))

sys1.exit()

res = compete_with_random(current_network, 500, False)