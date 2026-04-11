from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from horizon.models.blocks import DiTBlock, FinalLayer, timestep_embedding
from horizon.models.conditioning import ConditioningFusion
from horizon.utils.shapes import expect_last_dim, expect_rank


@dataclass
class ActionDiTConfig:
    action_dim: int
    action_chunk: int
    proprio_dim: int
    vis_feature_dim: int
    model_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    dropout: float
    text_vocab_size: int
    max_text_tokens: int
    use_proprio: bool = True
    use_task_text: bool = True


class ActionDiT(nn.Module):
    def __init__(self, config: ActionDiTConfig) -> None:
        super().__init__()
        self.config = config
        self.action_proj = nn.Linear(config.action_dim, config.model_dim)
        self.action_pos_embed = nn.Parameter(torch.randn(1, config.action_chunk, config.model_dim) * 0.02)
        self.fusion = ConditioningFusion(
            vis_feature_dim=config.vis_feature_dim,
            proprio_dim=config.proprio_dim,
            model_dim=config.model_dim,
            text_vocab_size=config.text_vocab_size,
            max_text_tokens=config.max_text_tokens,
            use_proprio=config.use_proprio,
            use_task_text=config.use_task_text,
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(config.model_dim, config.num_heads, config.mlp_ratio, config.dropout)
                for _ in range(config.depth)
            ]
        )
        self.final_layer = FinalLayer(config.model_dim, config.action_dim)

    def forward(self, noisy_actions: torch.Tensor, timesteps: torch.Tensor, conditioning: dict[str, Any]) -> torch.Tensor:
        expect_rank(noisy_actions, 3, "noisy_actions")
        expect_last_dim(noisy_actions, self.config.action_dim, "noisy_actions")

        fused = self.fusion(conditioning)
        action_tokens = self.action_proj(noisy_actions) + self.action_pos_embed[:, : noisy_actions.shape[1]]
        sequence = torch.cat([fused.context_tokens, action_tokens], dim=1)

        time_embed = timestep_embedding(timesteps, self.config.model_dim)
        cond_embed = self.time_mlp(time_embed) + fused.global_cond

        action_mask = torch.ones(noisy_actions.shape[:2], dtype=torch.bool, device=noisy_actions.device)
        sequence_mask = torch.cat([fused.context_mask, action_mask], dim=1)
        key_padding_mask = ~sequence_mask

        hidden = sequence
        for block in self.blocks:
            hidden = block(hidden, cond_embed, key_padding_mask=key_padding_mask)

        action_hidden = hidden[:, -noisy_actions.shape[1] :]
        return self.final_layer(action_hidden, cond_embed)
