from __future__ import annotations

import torch
from torch import nn

from lerobot_policy_horizon.utils.shapes import expect_rank


class SimpleVisualEncoder(nn.Module):
    """Shared lightweight encoder used as a local Cosmos fallback.

    Input:
    - frames: [B, T, C, H, W]

    Output:
    - features: [B, T, D]
    """

    def __init__(self, in_channels: int, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_proj = nn.Linear(128, feature_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        expect_rank(frames, 5, "frames")
        batch_size, steps, channels, height, width = frames.shape
        flat = frames.reshape(batch_size * steps, channels, height, width)
        encoded = self.encoder(flat).flatten(start_dim=1)
        features = self.out_proj(encoded)
        return features.view(batch_size, steps, self.feature_dim)
