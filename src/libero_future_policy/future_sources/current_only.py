from __future__ import annotations

from typing import Any

import torch

from libero_future_policy.future_sources.base import FUTURE_SOURCE_IDS, FutureConditionOutput, FutureSource


class CurrentOnlyFutureSource(FutureSource):
    name = "current_only"

    def get_future_condition(
        self,
        batch: dict[str, Any],
        backbone: Any,
        mode: str,
        step: int,
        rng: torch.Generator | None = None,
    ) -> FutureConditionOutput:
        current_frames = batch["current_frames"]
        batch_size = current_frames.shape[0]
        future_window = batch["future_frames"].shape[1]
        device = current_frames.device
        future_vis = backbone.empty_future_features(batch_size, future_window, device)
        future_mask = torch.zeros(batch_size, future_window, dtype=torch.bool, device=device)
        future_source_id = torch.full(
            (batch_size,),
            FUTURE_SOURCE_IDS[self.name],
            dtype=torch.long,
            device=device,
        )
        return FutureConditionOutput(
            future_vis=future_vis,
            future_mask=future_mask,
            future_source_id=future_source_id,
        )
