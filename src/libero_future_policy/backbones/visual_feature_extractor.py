from __future__ import annotations

import torch
from torch import nn

from libero_future_policy.utils.shapes import expect_rank


class SimpleVisualFeatureExtractor(nn.Module):
    """Small shared visual encoder for current and future frame windows."""

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
        # frames: [B, T, C, H, W]
        expect_rank(frames, 5, "frames")
        batch_size, steps, channels, _, _ = frames.shape
        flat = frames.reshape(batch_size * steps, channels, frames.shape[-2], frames.shape[-1])
        encoded = self.encoder(flat).flatten(start_dim=1)
        features = self.out_proj(encoded)
        return features.view(batch_size, steps, self.feature_dim)
