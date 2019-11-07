BOARD_DIMS = (5, 5)
BOARD_SIZE = BOARD_DIMS[0]
KOMI = 0 #POTE
ACTIONS = BOARD_DIMS[0]*BOARD_DIMS[1] + 2
SHOW = False
PIT = False

NUM_GAMES_PER_ITERATION = 50
SIMULATIONS = 100
HISTORY = 2
NUM_GAMES_COMPETITION = 5
TEMP_THRESHOLD_MOVES = 10

TRAINING_STEPS_PER_ITERATION = 1000
NUMBER_OF_FILTERS = 32
NUMBER_OF_RESIDUAL_BLOCKS = 4
VALUE_HIDDEN = 32
BATCH_SIZE = 256

MOVE_CAP = (3*BOARD_DIMS[0]*BOARD_DIMS[1])

BUFF_GAMES = 1000
AVG_GAME_LENGTH = int(BOARD_DIMS[0] * BOARD_DIMS[1] * 2.5)
BUFFER_SIZE = BUFF_GAMES * AVG_GAME_LENGTH

C_PUCT = 1

#PENDING, POTI - rotations before passing to nnet, exploiting symmetry of the game
#PENDING - v_resign??

#same board could have multiple parents - think ply4 ?? permutations of same moves #POTI
