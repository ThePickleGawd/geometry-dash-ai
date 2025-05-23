# See README for more details

# Model
FRAME_STACK_SIZE = 4 # How many frames to stack in state, sent to model
COLOR_CHANNELS = 1

# Agent
DEFAULT_REWARD = -1
JUMP_PUNISHMENT = -100
DEATH_PUNISHMENT = -10000
CHECKPOINT_REWARD = 50

# TCP Client
COMMAND_PORT = 22222 # Sends commands like "step, restart"
FRAME_PORT = 22223 # Receives frame buffer
BASE_URL = "127.0.0.1"

# Train settings
SAVE_EPOCH = 500
RANDOM_SPAWN = False
SET_SPAWN = 1

# Hyperparameters
STEPS_BEFORE_TARGET_UPDATE = 1000
ACTION_DIM = 2       # jump / no jump

LR = 1e-3
GAMMA = 0.99
EPSILON_START = 1
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

BUFFER_SIZE = 10000
BATCH_SIZE = 16
DEATH_BATCH_SIZE = 0
