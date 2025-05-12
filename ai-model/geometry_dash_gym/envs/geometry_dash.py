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
        info = gdclient.send_command("step" + (" hold" if action==1 else ""))

        if "error" in info and info["error"] != "":
            print(f"Error: {info["error"]}")
            return None, None, None, None

        done = info["dead"]
        observation = None # Observation handled by tcp client
        reward = 0.0
        print(info)

        return observation, reward, done, info

    def reset(self):
        # Reset the level
        info = gdclient.send_command("reset")
        print(info)

        observation = None # Dummy
        return observation

    def close(self):
        gdclient.close()
