from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from libero_future_policy.conditioning.proprio_encoder import ProprioEncoder
from libero_future_policy.conditioning.text_encoder import SimpleTextEncoder
from libero_future_policy.future_sources.base import FUTURE_SOURCE_IDS
from libero_future_policy.utils.masking import masked_mean
from libero_future_policy.utils.shapes import expect_last_dim, expect_rank


@dataclass
class ConditioningOutput:
    context_tokens: torch.Tensor
    context_mask: torch.Tensor
    global_cond: torch.Tensor


class ConditioningFusion(nn.Module):
    """Fuse visual, proprio, text, and source-id conditioning into context tokens."""

    def __init__(
        self,
        vis_feature_dim: int,
        proprio_dim: int,
        model_dim: int,
        text_vocab_size: int,
        max_text_tokens: int,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.curr_proj = nn.Linear(vis_feature_dim, model_dim)
        self.future_proj = nn.Linear(vis_feature_dim, model_dim)
        self.proprio_encoder = ProprioEncoder(proprio_dim, model_dim)
        self.text_encoder = SimpleTextEncoder(text_vocab_size, model_dim, max_text_tokens)
        self.source_embed = nn.Embedding(len(FUTURE_SOURCE_IDS), model_dim)

        self.modality_embed = nn.ParameterDict(
            {
                "curr": nn.Parameter(torch.randn(1, 1, model_dim) * 0.02),
                "future": nn.Parameter(torch.randn(1, 1, model_dim) * 0.02),
                "proprio": nn.Parameter(torch.randn(1, 1, model_dim) * 0.02),
                "text": nn.Parameter(torch.randn(1, 1, model_dim) * 0.02),
                "source": nn.Parameter(torch.randn(1, 1, model_dim) * 0.02),
            }
        )

    def forward(self, conditioning: dict[str, Any]) -> ConditioningOutput:
        curr_vis = conditioning["curr_vis"]
        future_vis = conditioning["future_vis"]
        future_mask = conditioning.get("future_mask")
        future_source_id = conditioning["future_source_id"]
        proprio = conditioning.get("proprio")
        text = conditioning.get("text", [])

        expect_rank(curr_vis, 3, "curr_vis")
        expect_rank(future_vis, 3, "future_vis")
        expect_last_dim(curr_vis, self.curr_proj.in_features, "curr_vis")
        expect_last_dim(future_vis, self.future_proj.in_features, "future_vis")

        curr_tokens = self.curr_proj(curr_vis) + self.modality_embed["curr"]
        curr_mask = torch.ones(curr_tokens.shape[:2], dtype=torch.bool, device=curr_tokens.device)

        future_tokens = self.future_proj(future_vis) + self.modality_embed["future"]
        if future_mask is None:
            future_mask = torch.ones(future_tokens.shape[:2], dtype=torch.bool, device=future_tokens.device)
        future_tokens = future_tokens * future_mask.unsqueeze(-1)

        if proprio is None:
            proprio_tokens = torch.zeros(
                curr_vis.shape[0],
                0,
                self.model_dim,
                device=curr_vis.device,
            )
            proprio_mask = torch.zeros(curr_vis.shape[0], 0, dtype=torch.bool, device=curr_vis.device)
        else:
            expect_rank(proprio, 3, "proprio")
            proprio_tokens, proprio_mask = self.proprio_encoder(proprio)
            proprio_tokens = proprio_tokens.to(curr_vis.device) + self.modality_embed["proprio"]

        if isinstance(text, list) and len(text) > 0:
            text_tokens, text_mask = self.text_encoder(text, curr_vis.device)
            text_tokens = text_tokens + self.modality_embed["text"]
        else:
            text_tokens = torch.zeros(curr_vis.shape[0], 0, self.model_dim, device=curr_vis.device)
            text_mask = torch.zeros(curr_vis.shape[0], 0, dtype=torch.bool, device=curr_vis.device)

        source_token = self.source_embed(future_source_id).unsqueeze(1) + self.modality_embed["source"]
        source_mask = torch.ones(curr_vis.shape[0], 1, dtype=torch.bool, device=curr_vis.device)

        context_tokens = torch.cat(
            [curr_tokens, future_tokens, proprio_tokens, text_tokens, source_token],
            dim=1,
        )
        context_mask = torch.cat(
            [curr_mask, future_mask, proprio_mask, text_mask, source_mask],
            dim=1,
        )
        context_tokens = context_tokens * context_mask.unsqueeze(-1)
        global_cond = masked_mean(context_tokens, context_mask)
        return ConditioningOutput(
            context_tokens=context_tokens,
            context_mask=context_mask,
            global_cond=global_cond,
        )
