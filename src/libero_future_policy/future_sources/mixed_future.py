from __future__ import annotations

from typing import Any

import torch

from libero_future_policy.future_sources.base import FutureConditionOutput, FutureSource, index_batch


class MixedFutureSource(FutureSource):
    name = "mixed"

    def __init__(self, sources: dict[str, FutureSource], probabilities: dict[str, float]) -> None:
        if not sources:
            raise ValueError("MixedFutureSource requires at least one source.")
        self.sources = sources
        self.names = list(sources.keys())

        probs = torch.tensor([probabilities[name] for name in self.names], dtype=torch.float32)
        if (probs < 0).any():
            raise ValueError("Mix probabilities must be non-negative.")
        if float(probs.sum().item()) <= 0.0:
            raise ValueError("Mix probabilities must sum to a positive value.")
        self.probabilities = probs / probs.sum()

    def get_future_condition(
        self,
        batch: dict[str, Any],
        backbone: Any,
        mode: str,
        step: int,
        rng: torch.Generator | None = None,
    ) -> FutureConditionOutput:
        batch_size = batch["current_frames"].shape[0]
        source_choices = torch.multinomial(
            self.probabilities,
            num_samples=batch_size,
            replacement=True,
            generator=rng,
        )

        future_window = batch["future_frames"].shape[1]
        device = batch["current_frames"].device
        future_vis = backbone.empty_future_features(batch_size, future_window, device)
        future_mask = torch.zeros(batch_size, future_window, dtype=torch.bool, device=device)
        future_source_id = torch.zeros(batch_size, dtype=torch.long, device=device)

        usage: dict[str, int] = {}
        for source_index, name in enumerate(self.names):
            indices = (source_choices == source_index).nonzero(as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            usage[name] = int(indices.numel())
            sub_batch = index_batch(batch, indices)
            sub_output = self.sources[name].get_future_condition(
                batch=sub_batch,
                backbone=backbone,
                mode=mode,
                step=step,
                rng=rng,
            )
            future_vis.index_copy_(0, indices, sub_output.future_vis)
            future_mask.index_copy_(0, indices, sub_output.future_mask)
            future_source_id.index_copy_(0, indices, sub_output.future_source_id)

        return FutureConditionOutput(
            future_vis=future_vis,
            future_mask=future_mask,
            future_source_id=future_source_id,
            metadata={"source_usage": usage},
        )
