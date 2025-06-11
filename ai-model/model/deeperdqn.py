import torch
from torch import nn
import torch.nn.functional as F
from config import COLOR_CHANNELS, FRAME_STACK_SIZE
from .NoisyNet import NoisyLinear, LazyNoisyLinear
import matplotlib.pyplot as plt
import math
import numpy as np
import config


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(self.block(x) + x)


class DeeperDQNModel(nn.Module):
    def __init__(
        self, in_channels=COLOR_CHANNELS, stack_size=FRAME_STACK_SIZE, num_actions=2
    ):
        super().__init__()
        input_channels = in_channels * stack_size

        self.conv = nn.Sequential(
            nn.Conv2d(
                input_channels, 32, kernel_size=8, stride=4
            ),  # [B, C*T, H, W] → [B, 32, H', W']
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
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B, T * C, H, W)
        x = self.conv(x)
        x = self.fc(x)
        return x


# Vigyan's version, frontloaded the deepness
class DeeperDQNModelv2(nn.Module):
    def __init__(
        self, in_channels=COLOR_CHANNELS, stack_size=FRAME_STACK_SIZE, num_actions=2
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels * stack_size, 48, kernel_size=8, stride=4
            ),  # input: [B, C*T, H, W]
            nn.SiLU(),
            nn.Conv2d(48, 64, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.SiLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(512), nn.SiLU(), nn.Linear(512, num_actions)
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
    def __init__(
        self, in_channels=COLOR_CHANNELS, stack_size=FRAME_STACK_SIZE, num_actions=2
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels * stack_size, 48, kernel_size=8, stride=4
            ),  # input: [B, C*T, H, W]
            nn.SiLU(),
            nn.Conv2d(48, 64, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.SiLU(),
        )

        self.lazynoise = LazyNoisyLinear(512, std=0)
        self.noise = NoisyLinear(512, num_actions, std=0)

        self.fc = nn.Sequential(
            nn.Flatten(),
            self.lazynoise,
            # nn.LazyLinear(512),
            nn.SiLU(),
            # nn.Linear(512, num_actions)
            self.noise,
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


class smallDQN(nn.Module):
    def __init__(
        self, in_channels=COLOR_CHANNELS, stack_size=FRAME_STACK_SIZE, num_actions=2
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels * stack_size, 16, kernel_size=8, stride=4
            ),  # input: [B, C*T, H, W]
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.SiLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(256), nn.SiLU(), nn.Linear(256, num_actions)
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


class ActionDeeperDQNModelv2(nn.Module):
    def __init__(
        self, in_channels=COLOR_CHANNELS, stack_size=FRAME_STACK_SIZE, num_actions=2
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels * stack_size, 48, kernel_size=8, stride=4
            ),  # input: [B, C*T, H, W]
            nn.SiLU(),
            nn.Conv2d(48, 64, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.SiLU(),
        )

        self.fc = nn.Sequential(
            nn.Identity(), nn.LazyLinear(512), nn.SiLU(), nn.Linear(512, num_actions)
        )

        # FOR VISUALS
        self.batch_count = 0

    def forward(self, x, action):
        """
        Batch, Time frame, color channels, Height , width
        x: (B, T, C, H, W) -> reshape to (B, T*C, H, W)
        """
        B, T, C, H, W = x.shape
        x = x.view(B, T * C, H, W)

        # FOR VISUALS:
        if config.VISUALS:
            self.record_visuals(x)

        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.concat((x, action), dim=1)
        x = self.fc(x)
        return x

    # FOR VISUALS
    # num_channels is how many channels in each layer u want to record
    def record_visuals(self, x, num_channels=FRAME_STACK_SIZE):
        self.batch_count += 1
        if self.batch_count == 20:
            x2 = x
            for l, layer in enumerate(self.conv):
                self.record_layer(x2[0], l)
                x2 = layer(x2)

    def record_layer(self, x, layer):
        x = x.detach().cpu().numpy()  # shape: [C, H, W]

        C, H, W = x.shape
        cols = math.ceil(math.sqrt(C))
        rows = math.ceil(C / cols)

        # Create an empty canvas with white background
        border_px = math.ceil(0.1 * H)
        grid_h = rows * H + (rows + 1) * border_px
        grid_w = cols * W + (cols + 1) * border_px
        grid = np.ones((grid_h, grid_w))  # white background (1.0 for grayscale)

        for idx in range(C):
            row = idx // cols
            col = idx % cols
            top = row * H + (row + 1) * border_px
            left = col * W + (col + 1) * border_px
            grid[top : top + H, left : left + W] = x[idx]

        self.save_image(grid, "layer" + str(layer))

    def save_image(self, grid, filename):
        plt.imshow(grid, cmap="gray")
        plt.axis("off")
        plt.savefig(f"visuals/{filename}", bbox_inches="tight", pad_inches=0)
        plt.close()
