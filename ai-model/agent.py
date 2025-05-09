import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

from config import (
    STATE_DIM, ACTION_DIM, LR, GAMMA,
    EPSILON_START, EPSILON_DECAY, EPSILON_MIN,
    BUFFER_SIZE, BATCH_SIZE
)

class Agent:
    def __init__(self):
        pass