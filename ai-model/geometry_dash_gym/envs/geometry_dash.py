import gymnasium as gym
from gymnasium import spaces

from enum import Enum
import numpy as np

from tcpclient import GDClient

class Actions(Enum):
    idle = 0
    hold = 1

class GeometryDashEnv(gym.Env):
    def __init__(self, host='127.0.0.1', port=22222):
        super(GeometryDashEnv, self).__init__()

        # Connect to GD mod
        self.client = GDClient(host, port)
        self.client.connect()

        self.action_space = spaces.Discrete(2)

        # Placeholder observation space (e.g., dummy 1D vector)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # State
        self.holding = False

    def step(self, action):
        # Send action to the GD mod, only if it's different from the last one
        if action == 1 and not self.holding:
            self.client.send_command("hold")
            self.holding = True
        elif action == 0 and self.holding:
            self.client.send_command("release")
            self.holding = False

        observation = np.array([0.0])  # placeholder observation
        reward = 0.0
        done = False
        info = {}

        return observation, reward, done, info

    def reset(self):
        # Reset the level
        self.client.send_command("reset")

        observation = np.array([0.0])
        return observation

    def close(self):
        self.client.close()
