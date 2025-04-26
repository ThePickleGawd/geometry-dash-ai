from enum import Enum
import gymnasium as gym

class Actions(Enum):
    nopress = 0
    press = 1

class GeometryDashEnv(gym.Env):
    def __init__(self):
        # Initialize the environment
        pass

    def reset(self, seed=None, options=None):
        # Reset the environment to an initial state

        # TODO: Get mod to reset game

        pass
        # return observation, info

    def step(self, action):
        # Take an action in the environment

        # TODO: We need the mod to always freeze the game. Here, we go to next frame

        pass
        # return observation, reward, done, info

    def render(self, mode='human'):
        # Render the environment
        pass

    def close(self):
        # Close the environment
        pass