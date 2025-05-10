import gymnasium as gym
from gymnasium import spaces

from enum import Enum
import numpy as np

from tcp import gdclient

class Actions(Enum):
    idle = 0
    hold = 1

class GeometryDashEnv(gym.Env):
    def __init__(self):
        super(GeometryDashEnv, self).__init__()

        self.action_space = spaces.Discrete(2)

        # Placeholder observation space (e.g., dummy 1D vector)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # State
        self.holding = False

    def step(self, action):
        print("in step function")
        # Send action to the GD mod, only if it's different from the last one
        if action == 1 and not self.holding:
            gdclient.send_command("hold")
            print("send hold")
            self.holding = True
        elif action == 0 and self.holding:
            gdclient.send_command("release")
            print("send release")
            self.holding = False

        gdclient.send_command("step")
        print("send step")
        observation = np.array([0.0])  # placeholder observation
        reward = 0.0
        done = False
        info = {}

        return observation, reward, done, info

    def reset(self):
        # Reset the level
        gdclient.send_command("reset")

        observation = np.array([0.0])
        return observation

    def close(self):
        gdclient.close()
