from __future__ import annotations

import torch
from torch import nn


def flatten_cosmos_features_to_sequence(x: torch.Tensor) -> torch.Tensor:
    """Normalize Cosmos block outputs to rank-3 ``[B, T_tokens, D]`` for per-token projection.

    - ``[B, D]`` → ``[B, 1, D]``
    - ``[B, T, D]`` unchanged
    - ``[B, C, H, W]`` → flatten spatial to ``[B, H*W, C]``
    - ``[B, C, T, H, W]`` → flatten spatiotemporal to ``[B, T*H*W, C]``
    """
    if x.ndim == 3:
        return x
    if x.ndim == 2:
        return x.unsqueeze(1)
    if x.ndim == 4:
        b, c, h, w = x.shape
        return x.permute(0, 2, 3, 1).reshape(b, h * w, c).contiguous()
    if x.ndim == 5:
        b, c, t, h, w = x.shape
        return x.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c).contiguous()
    raise ValueError(f"Unsupported tensor rank for Cosmos latents: ndim={x.ndim}, shape={tuple(x.shape)}")


class VisualLatentProjection(nn.Module):
    """Maps raw Cosmos token/grid features to ``vis_feature_dim`` for ``ConditioningFusion``.

    Used by ``HorizonDiTPolicy`` so ``CosmosAdapter`` stays free of learned projections.
    """

    def __init__(
        self,
        encode_in_dim: int,
        hidden_in_dim: int,
        out_dim: int,
        *,
        use_latent_projection: bool = True,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.use_latent_projection = use_latent_projection
        if use_latent_projection:
            self.encode_to_vis = nn.Linear(encode_in_dim, out_dim)
            self.hidden_to_vis = nn.Linear(hidden_in_dim, out_dim)
        else:
            self.encode_to_vis = nn.Identity()
            self.hidden_to_vis = nn.Identity()

    def forward_encode(self, x: torch.Tensor) -> torch.Tensor:
        x = flatten_cosmos_features_to_sequence(x)
        if not self.use_latent_projection:
            return x
        return self.encode_to_vis(x)

    def forward_hidden(self, x: torch.Tensor) -> torch.Tensor:
        x = flatten_cosmos_features_to_sequence(x)
        if not self.use_latent_projection:
            return x
        return self.hidden_to_vis(x)
