import torch
from torch import nn
import torch.nn.functional as F
from config import COLOR_CHANNELS, FRAME_STACK_SIZE
from .NoisyNet import NoisyLinear,LazyNoisyLinear

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(self.block(x) + x)

class DeeperDQNModel(nn.Module):
    def __init__(self, in_channels=COLOR_CHANNELS, stack_size=FRAME_STACK_SIZE, num_actions=2):
        super().__init__()
        input_channels = in_channels * stack_size

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),  # [B, C*T, H, W] → [B, 32, H', W']
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.SiLU(),
            ResidualBlock(64),
            ResidualBlock(64),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B, T * C, H, W)
        x = self.conv(x)
        x = self.fc(x)
        return x

# Vigyan's version, frontloaded the deepness
class DeeperDQNModelv2(nn.Module):
    def __init__(self, in_channels=COLOR_CHANNELS, stack_size=FRAME_STACK_SIZE, num_actions=2):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * stack_size, 48, kernel_size=8, stride=4),  # input: [B, C*T, H, W]
            nn.SiLU(),
            nn.Conv2d(48,64, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2),
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

class NoisyDeeperDQNModelv2(nn.Module):
    def __init__(self, in_channels=COLOR_CHANNELS, stack_size=FRAME_STACK_SIZE, num_actions=2):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * stack_size, 48, kernel_size=8, stride=4),  # input: [B, C*T, H, W]
            nn.SiLU(),
            nn.Conv2d(48,64, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.SiLU(),
        )

        self.lazynoise = LazyNoisyLinear(512,std=0)
        self.noise = NoisyLinear(512,num_actions,std=0)

        self.fc = nn.Sequential(
            nn.Flatten(),
            self.lazynoise,
            # nn.LazyLinear(512),
            nn.SiLU(),
            # nn.Linear(512, num_actions)
            self.noise
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
    
    def reset_noise(self):
        self.lazynoise.reset_noise()
        self.noise.reset_noise()