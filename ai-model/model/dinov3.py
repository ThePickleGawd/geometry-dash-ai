import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

import config


class DinoV3ExpertsModel(nn.Module):
    """MoE-style policy head driven by a frozen DINOv3 vision backbone."""

    def __init__(
        self,
        is_train: bool = False,
        num_actions: int = 2,
        model_name: str | None = None,
    ) -> None:
        super().__init__()
        self.is_train = is_train
        self.model_name = model_name or config.DINOV3_MODEL_ID

        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.backbone = AutoModel.from_pretrained(self.model_name)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.backbone.config, "embed_dim", None)
        if hidden_size is None:
            raise ValueError("Unable to determine hidden size from the DINOv3 configuration.")
        self.hidden_size = hidden_size

        if isinstance(self.image_processor.size, dict):
            if "height" in self.image_processor.size and "width" in self.image_processor.size:
                target_height = self.image_processor.size["height"]
                target_width = self.image_processor.size["width"]
            else:
                shortest = self.image_processor.size.get("shortest_edge", 224)
                target_height = target_width = shortest
        else:
            target_height = target_width = int(self.image_processor.size)
        self.target_size = (target_height, target_width)

        image_mean = torch.tensor(self.image_processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        image_std = torch.tensor(self.image_processor.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean, persistent=False)
        self.register_buffer("image_std", image_std, persistent=False)

        self.ship_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

        self.ship_fc = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.SiLU(),
            nn.Linear(512, num_actions),
        )

        self.cube_fc = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.SiLU(),
            nn.Linear(512, num_actions),
        )

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor, is_ship: torch.Tensor | None = None):
        """
        Args:
            x: Tensor of shape (B, T, C, H, W)
            is_ship: Optional mask (B,)
        Returns:
            tuple(pred_actions_q, pred_is_ships)
        """

        features = self._extract_features(x)

        pred_is_ships = torch.sigmoid(self.ship_classifier(features))
        ship_logits = self.ship_fc(features)
        cube_logits = self.cube_fc(features)

        if self.is_train and is_ship is not None:
            mask = is_ship.view(-1).bool()
        else:
            mask = pred_is_ships.view(-1) > 0.5

        pred_actions_q = torch.empty_like(ship_logits)
        pred_actions_q[mask] = ship_logits[mask]
        pred_actions_q[~mask] = cube_logits[~mask]

        return pred_actions_q, pred_is_ships

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # Use the most recent frame and broadcast to RGB if needed.
        latest_frame = x[:, -1]  # (B, C, H, W)
        if latest_frame.shape[1] == 1:
            latest_frame = latest_frame.repeat(1, 3, 1, 1)
        elif latest_frame.shape[1] > 3:
            latest_frame = latest_frame[:, :3]

        pixel_values = F.interpolate(
            latest_frame,
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        )
        pixel_values = (pixel_values - self.image_mean) / self.image_std

        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                features = outputs.pooler_output
            else:
                features = outputs.last_hidden_state.mean(dim=1)

        return features
