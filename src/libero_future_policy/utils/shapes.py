from __future__ import annotations

import torch


def expect_rank(tensor: torch.Tensor, rank: int, name: str) -> None:
    if tensor.ndim != rank:
        raise AssertionError(f"{name} must have rank {rank}, got shape {tuple(tensor.shape)}")


def expect_last_dim(tensor: torch.Tensor, size: int, name: str) -> None:
    if tensor.shape[-1] != size:
        raise AssertionError(f"{name} last dim must be {size}, got shape {tuple(tensor.shape)}")
