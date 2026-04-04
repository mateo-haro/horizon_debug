from __future__ import annotations

import hashlib

import torch
from torch import nn


class SimpleTextEncoder(nn.Module):
    """Deterministic hash-based text encoder with token outputs."""

    def __init__(self, vocab_size: int, embed_dim: int, max_tokens: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_tokens = max_tokens
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

    def _token_to_id(self, token: str) -> int:
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % self.vocab_size

    def tokenize(self, texts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(texts)
        ids = torch.zeros(batch_size, self.max_tokens, dtype=torch.long, device=device)
        mask = torch.zeros(batch_size, self.max_tokens, dtype=torch.bool, device=device)
        for batch_index, text in enumerate(texts):
            tokens = text.lower().split()[: self.max_tokens]
            for token_index, token in enumerate(tokens):
                ids[batch_index, token_index] = self._token_to_id(token)
                mask[batch_index, token_index] = True
        return ids, mask

    def forward(self, texts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if len(texts) == 0:
            return (
                torch.zeros(0, 0, self.embed_dim, device=device),
                torch.zeros(0, 0, dtype=torch.bool, device=device),
            )

        token_ids, token_mask = self.tokenize(texts, device)
        token_embeddings = self.embedding(token_ids)
        token_embeddings = self.output_proj(token_embeddings)
        token_embeddings = token_embeddings * token_mask.unsqueeze(-1)
        return token_embeddings, token_mask
