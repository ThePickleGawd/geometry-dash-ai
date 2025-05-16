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
        self.prePercent = 0

    def step(self, action):
        # INFO contains if dead and the percentage
        info = gdclient.send_command("step" + (" hold" if action==1 else ""))
        if "error" in info:
            print(f"Error: {info["error"]}")
            return None, None, None, None

        done = info["dead"]
        observation = None # Observation handled by tcp client
        reward = -1.0

        if done:
            reward = -1000

        return observation, reward, done, info

    def reset(self):
        # Reset the level
        info = gdclient.send_command("reset")
        print("done reset")
        print(info)

        observation = None # Dummy
        return observation

    def close(self):
        gdclient.close()
