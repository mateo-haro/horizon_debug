from __future__ import annotations

import torch
from torch import nn


class ProprioEncoder(nn.Module):
    """Lightweight sequence encoder for proprio windows."""

    def __init__(self, input_dim: int, model_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, proprio: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if proprio is None:
            device = torch.device("cpu")
            return (
                torch.zeros(1, 0, self.model_dim, device=device),
                torch.zeros(1, 0, dtype=torch.bool, device=device),
            )

        tokens = self.net(proprio)
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return tokens, mask
