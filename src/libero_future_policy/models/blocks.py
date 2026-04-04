from __future__ import annotations

import math

import torch
from torch import nn


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp = MLP(dim, mlp_ratio, dropout)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modulation(cond).chunk(6, dim=-1)

        attn_in = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(
            attn_in,
            attn_in,
            attn_in,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + gate_msa.unsqueeze(1) * attn_out

        mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in)

        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim),
        )
        self.linear = nn.Linear(dim, out_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(cond).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)
