import torch
from torch import nn
import torch.nn.functional as F
from config import COLOR_CHANNELS, FRAME_STACK_SIZE

class ExpertsModel(nn.Module):
    def __init__(self, in_channels=COLOR_CHANNELS, stack_size=FRAME_STACK_SIZE, num_actions=2):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels * stack_size, 32, kernel_size=8, stride=4),  # input: [B, C*T, H, W]
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.SiLU(),
        )

        self.ship_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),       # [B, C, H, W] -> [B, C, 1, 1]
            nn.Flatten(),                  # [B, C, 1, 1] -> [B, C]
            nn.LazyLinear(128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

        self.cube_fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.SiLU(),
            nn.Linear(512, num_actions)
        )

        self.ship_fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.SiLU(),
            nn.Linear(512, num_actions)
        )
        
    def forward(self, x, is_ship=None, is_train=False):
        """
        x: (B, T, C, H, W) -> reshape to (B, T*C, H, W)
        Returns:
            pred_action: (B, num_actions)
            pred_is_ship: (B, 1)
        """
        B, T, C, H, W = x.shape
        x = x.view(B, T * C, H, W)
        features = self.backbone(x)

        pred_is_ship = torch.sigmoid(self.ship_classifier(features))  # (B, 1)

        # Get logits from both experts
        ship_logits = self.ship_fc(features)  # (B, num_actions)
        cube_logits = self.cube_fc(features)  # (B, num_actions)

        if is_train:
            use_ship_expert = is_ship.view(-1, 1).float()  # (B, 1)
        else:
            use_ship_expert = (pred_is_ship > 0.5).float()  # (B, 1)

        # Select logits per sample using broadcasting
        pred_actions_q = use_ship_expert * ship_logits + (1 - use_ship_expert) * cube_logits

        return pred_actions_q, pred_is_ship

