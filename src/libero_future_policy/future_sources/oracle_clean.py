from __future__ import annotations

from typing import Any

import torch

from libero_future_policy.future_sources.base import FUTURE_SOURCE_IDS, FutureConditionOutput, FutureSource


class OracleCleanFutureSource(FutureSource):
    name = "oracle_clean"

    def get_future_condition(
        self,
        batch: dict[str, Any],
        backbone: Any,
        mode: str,
        step: int,
        rng: torch.Generator | None = None,
    ) -> FutureConditionOutput:
        future_frames = batch["future_frames"]
        future_vis = backbone.encode_future_frames(future_frames)
        future_mask = torch.ones(
            future_vis.shape[:2],
            dtype=torch.bool,
            device=future_vis.device,
        )
        future_source_id = torch.full(
            (future_vis.shape[0],),
            FUTURE_SOURCE_IDS[self.name],
            dtype=torch.long,
            device=future_vis.device,
        )
        return FutureConditionOutput(
            future_vis=future_vis,
            future_mask=future_mask,
            future_source_id=future_source_id,
        )
