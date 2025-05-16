import torch
from torch import nn
import torch.nn.functional as F

import config
from config import FRAME_STACK_SIZE, COLOR

class DQNModel(nn.Module):
    def __init__(self, in_channels=COLOR, stack_size=FRAME_STACK_SIZE, num_actions=2):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * stack_size, 32, kernel_size=8, stride=4),  # input: [B, C*T, H, W]
            nn.SiLU(),
            # nn.BatchNorm2d(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.SiLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.SiLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        """
        Batch, Time frame, color channels, Height , width
        x: (B, T, C, H, W) -> reshape to (B, T*C, H, W)
        """
        B, T, C, H, W = x.shape
        x = x.view(B, T * C, H, W)
        x = self.conv(x)
        x = self.fc(x)
        return x
