import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import math
import config
import cv2
from torch.distributions import Categorical
from model.dqnexperts import ExpertsModel, ExpertsFromDeeperDQNModelv2, ExpertsModelV2

from config import (
    ACTION_DIM,
    LR,
    GAMMA,
    EPSILON_START,
    EPSILON_DECAY,
    EPSILON_MIN,
    BUFFER_SIZE,
    BATCH_SIZE,
    DEATH_BATCH_SIZE,
    COLOR_CHANNELS,
    FRAME_STACK_SIZE,
)


class AgentMoE_GRPO:
    def __init__(self, model):
        self.action_dim = ACTION_DIM

        assert (
            type(model) == ExpertsModel
            or type(model) == ExpertsFromDeeperDQNModelv2
            or type(model) == ExpertsModelV2
        ), "Only use the Experts Model"

        self.device = next(model.parameters()).device
        self.model = model
        self.target_model = type(model)().to(self.device)
        self.target_model.load_state_dict(model.state_dict())

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        self.criterion_ship_classify = nn.BCELoss()
        self.cls_lambda = 0.1  # How much to weight classification loss
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.adv_norm_eps = 1e-6
        self.last_log_prob = None

        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN

        self.death_replay_buffer = deque(maxlen=config.NSTEP)
        self.replay_buffer = deque(maxlen=BUFFER_SIZE)

    def act(self, state, is_ship=None):
        state_device = state.to(self.device)
        ship_tensor = (
            torch.tensor([is_ship], device=self.device, dtype=torch.bool)
            if is_ship is not None
            else torch.tensor([False], device=self.device)
        )

        with torch.no_grad():
            action_logits, _ = self.model(state_device, ship_tensor)
            policy_dist = Categorical(logits=action_logits)
            policy_probs = torch.softmax(action_logits, dim=-1).squeeze(0)

            if self.epsilon > 0 and random.random() < self.epsilon:
                action_tensor = torch.randint(
                    0, self.action_dim, (1,), device=self.device
                )
            else:
                action_tensor = policy_dist.sample()

            action = action_tensor.item()
            behavior_prob = (
                (1.0 - self.epsilon) * policy_probs[action]
                + self.epsilon / self.action_dim
            )
            log_prob = torch.log(behavior_prob.clamp(min=1e-8))

        self.last_log_prob = log_prob.cpu().item()
        return action

    def remember(self, state, action, reward, next_state, is_ship, done):
        if self.last_log_prob is None:
            raise RuntimeError(
                "Action log probability missing. Ensure `act` is called before `remember`."
            )

        transition = (
            state.detach().clone(),
            action,
            reward,
            next_state.detach().clone(),
            is_ship,
            done,
            self.last_log_prob,
        )
        self.replay_buffer.append(transition)
        self.last_log_prob = None

    def on_death(self):
        if len(self.replay_buffer) < 5:
            return

        # Set the last 5 frames to death penalty
        last_reward = self.replay_buffer[-1][2]
        for i in range(-5, 0):
            s, a, _, ns, is_ship, d, log_prob = self.replay_buffer[i]
            self.replay_buffer[i] = (s, a, last_reward, ns, is_ship, d, log_prob)
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

        states, actions, rewards, _, is_ships, _, log_probs = zip(*batch)

        states = torch.cat([s.float() for s in states], dim=0).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        is_ships_tensor = torch.tensor(is_ships, dtype=torch.bool, device=self.device)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32, device=self.device)

        action_logits, pred_is_ships = self.model(states, is_ships_tensor)
        pred_is_ships = pred_is_ships.squeeze(-1)

        dist = Categorical(logits=action_logits)
        new_log_probs = dist.log_prob(actions)
        ratios = torch.exp(new_log_probs - old_log_probs)

        advantages = torch.zeros_like(rewards)
        ship_mask = is_ships_tensor
        for mask in (ship_mask, ~ship_mask):
            if mask.any():
                group_rewards = rewards[mask]
                group_adv = group_rewards - group_rewards.mean()
                group_std = group_adv.std(unbiased=False)
                if group_std > self.adv_norm_eps:
                    group_adv = group_adv / (group_std + self.adv_norm_eps)
                advantages[mask] = group_adv

        if torch.allclose(advantages, torch.zeros_like(advantages)):
            advantages = rewards - rewards.mean()
            std = advantages.std(unbiased=False)
            if std > self.adv_norm_eps:
                advantages = advantages / (std + self.adv_norm_eps)

        clipped_ratios = torch.clamp(
            ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon
        )
        surrogate_1 = ratios * advantages
        surrogate_2 = clipped_ratios * advantages
        policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

        cls_loss = self.criterion_ship_classify(
            pred_is_ships, is_ships_tensor.float()
        )
        entropy_loss = -self.entropy_coef * dist.entropy().mean()

        loss = policy_loss + self.cls_lambda * cls_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Return ship pred acc
        pred_labels = (pred_is_ships > 0.5).float()
        correct = torch.count_nonzero(pred_labels == is_ships_tensor.float())
        return correct / pred_labels.shape[0]

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
