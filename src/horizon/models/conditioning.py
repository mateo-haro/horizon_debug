from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from horizon.future_sources.base import FUTURE_SOURCE_IDS
from horizon.utils.masking import masked_mean
from horizon.utils.shapes import expect_last_dim, expect_rank


class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_tokens: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_tokens = max_tokens
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.output_proj = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim))

    def _token_to_id(self, token: str) -> int:
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % self.vocab_size

    def forward(self, texts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(texts)
        ids = torch.zeros(batch_size, self.max_tokens, dtype=torch.long, device=device)
        mask = torch.zeros(batch_size, self.max_tokens, dtype=torch.bool, device=device)
        for batch_index, text in enumerate(texts):
            for token_index, token in enumerate(text.lower().split()[: self.max_tokens]):
                ids[batch_index, token_index] = self._token_to_id(token)
                mask[batch_index, token_index] = True
        embeddings = self.output_proj(self.embedding(ids))
        embeddings = embeddings * mask.unsqueeze(-1)
        return embeddings, mask


class ProprioEncoder(nn.Module):
    def __init__(self, input_dim: int, model_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim))

    def forward(self, proprio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.net(proprio)
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return tokens, mask


@dataclass
class ConditioningOutput:
    context_tokens: torch.Tensor
    context_mask: torch.Tensor
    global_cond: torch.Tensor


class ConditioningFusion(nn.Module):
    def __init__(
        self,
        vis_feature_dim: int,
        proprio_dim: int,
        model_dim: int,
        text_vocab_size: int,
        max_text_tokens: int,
        use_proprio: bool,
        use_task_text: bool,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.use_proprio = use_proprio
        self.use_task_text = use_task_text
        self.curr_proj = nn.Linear(vis_feature_dim, model_dim)
        self.future_proj = nn.Linear(vis_feature_dim, model_dim)
        self.proprio_encoder = ProprioEncoder(max(1, proprio_dim), model_dim)
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
        future_mask = conditioning["future_mask"]
        future_source_id = conditioning["future_source_id"]
        proprio = conditioning.get("proprio")
        texts = conditioning.get("text", [])

        expect_rank(curr_vis, 3, "curr_vis")
        expect_rank(future_vis, 3, "future_vis")
        expect_last_dim(curr_vis, self.curr_proj.in_features, "curr_vis")
        expect_last_dim(future_vis, self.future_proj.in_features, "future_vis")

        curr_tokens = self.curr_proj(curr_vis) + self.modality_embed["curr"]
        curr_mask = torch.ones(curr_tokens.shape[:2], dtype=torch.bool, device=curr_tokens.device)

        future_tokens = self.future_proj(future_vis) + self.modality_embed["future"]
        future_tokens = future_tokens * future_mask.unsqueeze(-1)

        if self.use_proprio and proprio is not None:
            proprio_tokens, proprio_mask = self.proprio_encoder(proprio)
            proprio_tokens = proprio_tokens + self.modality_embed["proprio"]
        else:
            proprio_tokens = torch.zeros(curr_vis.shape[0], 0, self.model_dim, device=curr_vis.device)
            proprio_mask = torch.zeros(curr_vis.shape[0], 0, dtype=torch.bool, device=curr_vis.device)

        if self.use_task_text and texts:
            text_tokens, text_mask = self.text_encoder(texts, curr_vis.device)
            text_tokens = text_tokens + self.modality_embed["text"]
        else:
            text_tokens = torch.zeros(curr_vis.shape[0], 0, self.model_dim, device=curr_vis.device)
            text_mask = torch.zeros(curr_vis.shape[0], 0, dtype=torch.bool, device=curr_vis.device)

        source_token = self.source_embed(future_source_id).unsqueeze(1) + self.modality_embed["source"]
        source_mask = torch.ones(curr_vis.shape[0], 1, dtype=torch.bool, device=curr_vis.device)

        context_tokens = torch.cat([curr_tokens, future_tokens, proprio_tokens, text_tokens, source_token], dim=1)
        context_mask = torch.cat([curr_mask, future_mask, proprio_mask, text_mask, source_mask], dim=1)
        context_tokens = context_tokens * context_mask.unsqueeze(-1)
        global_cond = masked_mean(context_tokens, context_mask)
        return ConditioningOutput(context_tokens=context_tokens, context_mask=context_mask, global_cond=global_cond)
