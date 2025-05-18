import torch
from torch import nn
import torch.nn.functional as F
from config import COLOR_CHANNELS, FRAME_STACK_SIZE

class DUEL_DQNModel(nn.Module):
    def __init__(self, in_channels=COLOR_CHANNELS, stack_size=FRAME_STACK_SIZE, num_actions=2):
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

        self.flatten = nn.Flatten()

        self.value_model = nn.Sequential(
            nn.LazyLinear(512),
            nn.SiLU(),
            nn.Linear(512, 1)  
        )

        # Advantage stream
        self.advantage_model = nn.Sequential(
            nn.LazyLinear(512),
            nn.SiLU(),
            nn.Linear(512, num_actions)  
        )

    def forward(self, x):
        """
        x: (B, T, C, H, W) — batch of stacked frames
        """
        B, T, C, H, W = x.shape
        x = x.view(B, T * C, H, W)  # Merge time and channel dims
        x = self.conv(x)
        x = self.flatten(x)

        value = self.value_model(x)           
        advantage = self.advantage_model(x)    

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True)) 
        return q_values