import gymnasium as gym
from gymnasium import spaces
import random
from enum import Enum
import numpy as np
import config
import math
from collections import deque

from tcp import gdclient

class Actions(Enum):
    idle = 0
    hold = 1

class GeometryDashEnv(gym.Env):
    def __init__(self):
        super(GeometryDashEnv, self).__init__()

        self.action_space = spaces.Discrete(2)
        #FOR NEWSTATE
        # self.percentCount = deque(maxlen=config.PERCENT_BUFFER_SIZE)

        # Placeholder observation space (e.g., dummy 1D vector)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # State
        self.holding = False
        self.prePercent = 0

    def step(self, action, start_percent=1):
        # INFO contains if dead and the percentage
        info = gdclient.send_command("step" + (" hold" if action==1 else ""))
        if "error" in info:
            print(f"Error: {info["error"]}")
            return None, None, None, None

        #update percentCount buffer for NEWSTATE
        # if not self.percentCount[-1][int(info['percent'])] and start_percent==config.SET_SPAWN:
        #         self.percentCount[-1][int(info['percent'])] = True

        done = info["dead"]
        observation = None # Observation handled by tcp client
        reward = config.DEFAULT_REWARD
        #Squared reward of percent done
        # reward = config.DEFAULT_REWARD + (info['percent']-start_percent)**2
        
        #IF STATEMENT TO REMOVE JUMP PUNISHMENT FROM SHIP
        # if (action==1 and info['percent']<86 and not ((info['percent']>30)and(info['percent']<46.79))):
        if action == 1:
            reward = config.JUMP_PUNISHMENT
        if (info['percent'] > self.prePercent and (info['percent']%3) < (self.prePercent%3)):
            reward = config.CHECKPOINT_REWARD

        if done:
            reward = config.DEATH_PUNISHMENT
            print("died")

        #for NEWSTATE
        # totalvisits= 0
        # for i in range(len(self.percentCount)):
        #     totalvisits += self.percentCount[i][int(info['percent'])]


        #Code for skipping ship
        # if (info['percent']>28 and info['percent']<46):
        #     reward += config.BEATING_LEVEL/2
        #     self.reset(47)
        
        # if (info['percent']>85.9):
        #     done = True
        #     reward += config.BEATING_LEVEL
        #     print('BEAT LEVEL!!!! (cube part)')
            
        #Code for skipping cube
        # if (info['percent']>46.79 and info['percent']<85):
        #     reward += config.BEATING_LEVEL/2
        #     self.reset(86)
        
        if (info['percent']>99):
            done = True
            reward += config.BEATING_LEVEL
            print('BEAT LEVEL!!!!')


        self.prePercent = info['percent']
        return observation, reward, done, info

    def reset(self, percent=1):
        # Reset the level
        info = gdclient.send_command(f"reset {percent}")
        observation = None # Dummy
        return observation

    def close(self):
        gdclient.close()
