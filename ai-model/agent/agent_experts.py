import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import math
import config
from model.dqnexperts import ExpertsModel

from config import (
    ACTION_DIM, LR, GAMMA,
    EPSILON_START, EPSILON_DECAY, EPSILON_MIN,
    BUFFER_SIZE, BATCH_SIZE, DEATH_BATCH_SIZE
)

class AgentExperts:
    def __init__(self, model):
        self.action_dim = ACTION_DIM

        assert type(model) == ExpertsModel, "Only use the Experts Model"
        
        self.device = next(model.parameters()).device
        self.model = model
        self.target_model = type(model)().to(self.device)
        self.target_model.load_state_dict(model.state_dict())

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        self.criterion_ship_classify = nn.BCELoss()
        self.cls_lambda = 0.2 # How much to weight classification loss

        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN

        self.temp_nstep_buffer = deque(maxlen=config.NSTEP)
        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.batch_size = BATCH_SIZE

    def act(self, state, is_ship):
        state = state.to(self.device)

        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            q_values, _ = self.model(state.to(self.device), torch.tensor([is_ship], device=self.device))
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, is_ship, done):
        self.replay_buffer.append((state, action, reward, next_state, is_ship, done))

    def save_death_replay(self):
        # Must be called on death. Will save the last few frames. Ignore frames right when we die
        last_5 = list(self.replay_buffer)[-20:-10]
        for t in last_5:
            self.death_replay_buffer.append(t)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)

        # We want to enforce some death memories
        # if len(self.death_replay_buffer) > self.death_batch_size:
        #     batch.extend(random.sample(self.death_replay_buffer, self.death_batch_size))

        states, actions, rewards, next_states, is_ships, dones = zip(*batch)

        states = torch.cat(states, dim=0).float().to(self.device)
        next_states = torch.cat(next_states, dim=0).float().to(self.device)
        actions = torch.tensor(actions).long().unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        dones = torch.tensor(dones).float().to(self.device)
        is_ships = torch.tensor(is_ships).bool().to(self.device)

        pred_actions_q, pred_is_ships = self.model(states, is_ships)
        next_actions_q, _ = self.target_model(next_states, is_ships) # Assuming is ships doesn't change
        pred_is_ships = pred_is_ships.squeeze(-1)

        curr_q = pred_actions_q.gather(1, actions).squeeze() # Get the right q pred
        next_q = next_actions_q.max(1)[0].detach() # Get the best next q

        expected_q = rewards + (self.gamma) * next_q * (1 - dones)
        
        # L = (1-lambda) * MSELoss + (lambda) * BCELoss
        loss = (1 - self.cls_lambda) * self.criterion(curr_q, expected_q) + self.cls_lambda * self.criterion_ship_classify(pred_is_ships, is_ships)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
