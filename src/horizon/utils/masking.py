from __future__ import annotations

import torch


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float().unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (tokens * weights).sum(dim=1) / denom
