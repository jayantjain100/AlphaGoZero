BOARD_DIMS = (5,5)
BOARD_SIZE = BOARD_DIMS[0]
KOMI = 0.5 #POTE, print nhi ho raha bas
ACTIONS = BOARD_DIMS[0]*BOARD_DIMS[1] + 2
SHOW = False
PIT = False

FREQUENCY_AGAINST_RANDOM = 50
GAMES_AGAINST_RANDOM = 1
FREQUENCY_MODEL_SAVING = 10
# NUM_GAMES_PER_ITERATION = 50
NUM_GAMES_PER_ITERATION = 25
SIMULATIONS = 100
HISTORY = 1
NUM_GAMES_COMPETITION = 5
TEMP_THRESHOLD_MOVES = 10

TRAINING_STEPS_PER_ITERATION = 200
NUMBER_OF_FILTERS = 32
NUMBER_OF_RESIDUAL_BLOCKS = 4
VALUE_HIDDEN = 32
BATCH_SIZE = 32

MOVE_CAP = (3*BOARD_DIMS[0]*BOARD_DIMS[1])

BUFF_GAMES = 10000
AVG_GAME_LENGTH = int(BOARD_DIMS[0] * BOARD_DIMS[1] * 2.5)
BUFFER_SIZE = BUFF_GAMES * AVG_GAME_LENGTH

C_PUCT = 1

PROCESSES = 1
#PENDING, POTI - rotations before passing to nnet, exploiting symmetry of the game
#PENDING - v_resign??

#same board could have multiple parents - think ply4 ?? permutations of same moves #POTI
