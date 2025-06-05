import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import math
import config

from config import (
    ACTION_DIM, LR, GAMMA,
    EPSILON_START, EPSILON_DECAY, EPSILON_MIN,
    BUFFER_SIZE, BATCH_SIZE, DEATH_BATCH_SIZE
)

class Agent:
    def __init__(self, model, start_ep=0):
        self.action_dim = ACTION_DIM
        
        self.device = next(model.parameters()).device
        self.model = model
        self.target_model = type(model)().to(self.device)
        self.target_model.load_state_dict(model.state_dict())

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN

        self.temp_nstep_buffer = deque(maxlen=config.NSTEP)
        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        # self.death_replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.batch_size = BATCH_SIZE
        # self.death_batch_size = DEATH_BATCH_SIZE

    def act(self, state):
        state = state.to(self.device)

        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            q_values = self.model(state.to(self.device))
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        
        #NSTEP BUFFER CODE
        # self.temp_nstep_buffer.append((state, action, reward, next_state, done,preaction))
        # if len(self.temp_nstep_buffer) == config.NSTEP:
        #     reward = 0
        #     currentstep = None
        #     isDone = False
        #     for i, step in enumerate(self.temp_nstep_buffer):
        #         reward += (config.GAMMA ** i) * step[2]
        #         currentstep = step
        #         if step[4]:
        #             isDone = True
        #             break
        #     self.replay_buffer.append((self.temp_nstep_buffer[-1][0],   \
        #         self.temp_nstep_buffer[-1][1],reward,currentstep[0],isDone,self.temp_nstep_buffer[-1][5]))

    # def save_death_replay(self):
    #     # Must be called on death. Will save the last few frames. Ignore frames right when we die
    #     last_5 = list(self.replay_buffer)[-20:-10]
    #     for t in last_5:
    #         self.death_replay_buffer.append(t)


    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)

        # We want to enforce some death memories
        # if len(self.death_replay_buffer) > self.death_batch_size:
        #     batch.extend(random.sample(self.death_replay_buffer, self.death_batch_size))

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states, dim=0).float().to(self.device)
        next_states = torch.cat(next_states, dim=0).float().to(self.device)
        actions = torch.tensor(actions).long().unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        dones = torch.tensor(dones).float().to(self.device)

        curr_q = self.model(states).gather(1, actions).squeeze()
        # next_q = self.target_model(next_states).max(1)[0].detach()

        #FOR DUELING(double?) DQN !
        next_actions = self.model(next_states).argmax(1, keepdim = True)
        next_q = self.target_model(next_states).gather(1, next_actions).squeeze().detach()


        # expected_q = rewards + (self.gamma**config.NSTEP) * next_q * (1 - dones)
        expected_q = rewards + (self.gamma) * next_q * (1 - dones)
        
        loss = self.criterion(curr_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
