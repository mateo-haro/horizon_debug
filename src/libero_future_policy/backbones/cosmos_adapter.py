from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from libero_future_policy.backbones.visual_feature_extractor import SimpleVisualFeatureExtractor
from libero_future_policy.utils.shapes import expect_rank


@dataclass
class CosmosAdapterConfig:
    feature_dim: int
    use_external_cosmos: bool = False
    external_cosmos_module: str | None = None
    freeze_backbone: bool = False
    image_channels: int = 3


class CosmosAdapter(nn.Module):
    """Repository-local boundary around any external Cosmos runtime.

    The rest of the repo should only depend on this adapter and never import
    Cosmos internals directly.
    """

    def __init__(self, config: CosmosAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_dim = config.feature_dim
        self.runtime: Any | None = None
        self.encoder = SimpleVisualFeatureExtractor(
            in_channels=config.image_channels,
            feature_dim=config.feature_dim,
        )

        if config.use_external_cosmos:
            self.runtime = self._try_load_external_runtime(config.external_cosmos_module)

        if config.freeze_backbone:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def _try_load_external_runtime(self, module_name: str | None) -> Any | None:
        if module_name is None:
            return None
        try:
            return importlib.import_module(module_name)
        except Exception:
            return None

    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        expect_rank(frames, 5, "frames")
        return self.encoder(frames)

    def encode_current_frames(self, frames: torch.Tensor) -> torch.Tensor:
        return self._encode_frames(frames)

    def encode_future_frames(self, frames: torch.Tensor) -> torch.Tensor:
        return self._encode_frames(frames)

    def maybe_generate_future_features(
        self,
        batch: dict[str, Any],
        num_samples: int = 1,
        step: int | None = None,
    ) -> torch.Tensor | None:
        """TODO: connect frozen Cosmos Predict2 generation here.

        Expected return shape:
        - single sample: [B, T_future, D_vis]
        - multiple stochastic samples can later be [B, K, T_future, D_vis]
        """

        if self.runtime is None:
            return None

        # TODO: Replace this with real Cosmos rollout/generation calls.
        # The adapter contract intentionally hides all external repo details.
        return None

    def maybe_extract_intermediate_features(
        self,
        frames: torch.Tensor,
        denoise_level: float | None = None,
    ) -> torch.Tensor | None:
        """TODO: return intermediate denoiser-state features when available."""

        if self.runtime is None:
            return None

        # TODO: call the external Cosmos model and expose intermediate hidden states.
        return None

    def empty_future_features(
        self,
        batch_size: int,
        future_window: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, future_window, self.feature_dim, device=device)
