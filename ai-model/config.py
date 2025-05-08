# See README for more details
CLIENT_PORT = 22222 # Sends commands like "step, restart"
SERVER_PORT = 22223 # Receives frame buffer

BASE_URL = "127.0.0.1"

# Hyperparameters
STATE_DIM = 84 * 84  # example flattened frame
ACTION_DIM = 2       # jump / no jump

LR = 1e-3
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

BUFFER_SIZE = 10000
BATCH_SIZE = 64
