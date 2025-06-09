import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import math
import config
import cv2
from model.dqnexperts import ExpertsModel, ExpertsFromDeeperDQNModelv2, ExpertsModelV2

from config import (
    ACTION_DIM, LR, GAMMA,
    EPSILON_START, EPSILON_DECAY, EPSILON_MIN,
    BUFFER_SIZE, BATCH_SIZE, DEATH_BATCH_SIZE,
    COLOR_CHANNELS, FRAME_STACK_SIZE
)

class AgentExperts:
    def __init__(self, model):
        self.action_dim = ACTION_DIM

        assert type(model) == ExpertsModel or type(model) == ExpertsFromDeeperDQNModelv2 or type(model) == ExpertsModelV2, "Only use the Experts Model"
        
        self.device = next(model.parameters()).device
        self.model = model
        self.target_model = type(model)().to(self.device)
        self.target_model.load_state_dict(model.state_dict())

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        self.criterion_ship_classify = nn.BCELoss()
        self.cls_lambda = 0.1 # How much to weight classification loss

        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN

        self.death_replay_buffer = deque(maxlen=config.NSTEP)
        self.replay_buffer = deque(maxlen=BUFFER_SIZE)

    def act(self, state, is_ship=None):
        state = state.to(self.device)

        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            q_values, _ = self.model(state.to(self.device), torch.tensor([is_ship], device=self.device))
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, is_ship, done):
        self.replay_buffer.append((state, action, reward, next_state, is_ship, done))

    def on_death(self):
        if len(self.replay_buffer) < 5:
            return

        # Set the last 5 frames to death penalty
        last_reward = self.replay_buffer[-1][2]
        for i in range(-5, 0):
            s, a, _, ns, is_ship, d = self.replay_buffer[i]
            self.replay_buffer[i] = (s, a, last_reward, ns, is_ship, d)
            self.death_replay_buffer.append(self.replay_buffer[i])

        # Visualize the death frame we saved
        # img = self.death_replay_buffer[-1][0][0, -1]  # take the most recent frame (C, H, W)
        # img_np = img.permute(1, 2, 0).cpu().numpy()  # CHW → HWC
        # img_np = (img_np * 255).astype('uint8')      # scale to [0, 255] if float32

        # cv2.imshow("test", img_np)
        # cv2.waitKey(0)

    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        batch = random.sample(self.replay_buffer, BATCH_SIZE)

        # We want to enforce some death memories
        if len(self.death_replay_buffer) > DEATH_BATCH_SIZE:
            batch.extend(random.sample(self.death_replay_buffer, DEATH_BATCH_SIZE))

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

        # Print accuracy
        pred_labels = (pred_is_ships > 0.5).float()  # or .int() if is_ships is int
        correct = torch.count_nonzero(pred_labels == is_ships)
        # print(f"Is ship: {correct}/{pred_labels.shape[0]} Accuracy: {correct / pred_labels.shape[0]:.2f}")

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

        # Return ship pred acc
        return correct/pred_labels.shape[0]

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
