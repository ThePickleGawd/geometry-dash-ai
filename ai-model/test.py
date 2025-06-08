import os
import time
import subprocess
import queue
import torch
from threading import Thread
from collections import deque
from torchvision.transforms import v2
from tqdm import tqdm
import cv2

from gym import GeometryDashEnv
from tcp import gdclient
from model import DQNModel, DUEL_DQNModel, DeeperDQNModel, DeeperDQNModelv2, NoisyDeeperDQNModelv2, smallDQN, ActionDeeperDQNModelv2
from agent import Agent
import config
import random
import math

device = torch.device('cpu')
model = DeeperDQNModelv2().to(device=device)
newModel = ActionDeeperDQNModelv2().to(device=device)

cp = torch.load("previousModels/HSV/clippedRewardv1cube/4803_70.7.pt",weights_only=False)
model_state = (cp["model_state"])

for name in model_state:
    print(name,model_state[name].shape)
newModel(torch.zeros(1,4,1,128,128),torch.zeros(1,2))
print('\n\n')
for name in newModel.state_dict():
    print(name,newModel.state_dict()[name].shape)

newParams = newModel.state_dict()
for name in model_state:
    if name != 'fc.1.weight':
        newModel.state_dict()[name].copy_(model_state[name])
    else:
        newModel.state_dict()[name].data[:,:256].copy_(model_state[name].data)

torch.save({
    "episode": cp['episode'],
    "model_state": newModel.state_dict(),
    "optimizer_state": cp['optimizer_state'],
    "time_alive": cp['time_alive'],
    "total_reward": cp['total_reward'],
    "epsilon":cp['epsilon'],
}, f'checkpoints/latest.pt')