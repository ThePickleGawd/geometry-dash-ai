import torch
from torch import nn
import torch.nn.functional as F

class DQNModel(nn.Module):
    def __init__(self, in_channels=3, num_actions=2):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),  # [32, 179, 318]
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),           # [64, 88, 158]
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),           # [64, 43, 78]
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
        x: (1, 3, 1270, 720)
        """

        x = self.conv(x)
        x = self.fc(x)
        return x
