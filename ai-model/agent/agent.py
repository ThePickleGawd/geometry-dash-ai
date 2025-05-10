import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

from config import (
    ACTION_DIM, LR, GAMMA,
    EPSILON_START, EPSILON_DECAY, EPSILON_MIN,
    BUFFER_SIZE, BATCH_SIZE
)

class Agent:
    def __init__(self, model):
        self.action_dim = ACTION_DIM
        self.model = model
        self.target_model = type(model)()
        self.target_model.load_state_dict(model.state_dict())

        self.optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN

        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.batch_size = BATCH_SIZE

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states, dim=0).float()
        next_states = torch.cat(next_states, dim=0).float()
        actions = torch.tensor(actions).long().unsqueeze(1)
        rewards = torch.tensor(rewards).float()
        dones = torch.tensor(dones).float()

        curr_q = self.model(states).gather(1, actions).squeeze()
        next_q = self.target_model(next_states).max(1)[0].detach()
        expected_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.criterion(curr_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
