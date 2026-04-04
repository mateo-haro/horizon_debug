from __future__ import annotations

from typing import Any

import torch

from libero_future_policy.future_sources.base import FUTURE_SOURCE_IDS, FutureConditionOutput, FutureSource


class GeneratedFutureSource(FutureSource):
    name = "generated"

    def get_future_condition(
        self,
        batch: dict[str, Any],
        backbone: Any,
        mode: str,
        step: int,
        rng: torch.Generator | None = None,
    ) -> FutureConditionOutput:
        generated = backbone.maybe_generate_future_features(batch=batch, num_samples=1, step=step)
        if generated is None:
            raise NotImplementedError(
                "Generated future conditioning is not connected yet. "
                "Implement CosmosAdapter.maybe_generate_future_features first."
            )

        future_mask = torch.ones(
            generated.shape[:2],
            dtype=torch.bool,
            device=generated.device,
        )
        future_source_id = torch.full(
            (generated.shape[0],),
            FUTURE_SOURCE_IDS[self.name],
            dtype=torch.long,
            device=generated.device,
        )
        return FutureConditionOutput(
            future_vis=generated,
            future_mask=future_mask,
            future_source_id=future_source_id,
        )
