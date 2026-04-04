from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from libero_future_policy.future_sources.base import FUTURE_SOURCE_IDS, FutureConditionOutput, FutureSource


@dataclass
class OracleNoisedConfig:
    noise_std: float = 0.2
    token_dropout_prob: float = 0.1


class OracleNoisedFutureSource(FutureSource):
    name = "oracle_noised"

    def __init__(self, config: OracleNoisedConfig) -> None:
        self.config = config

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

        alpha = torch.rand(
            future_vis.shape[0],
            1,
            1,
            device=future_vis.device,
            generator=rng,
        )
        noise = torch.randn(
            future_vis.shape,
            device=future_vis.device,
            dtype=future_vis.dtype,
            generator=rng,
        )
        noised = future_vis + alpha * self.config.noise_std * noise

        future_mask = torch.ones(
            future_vis.shape[:2],
            dtype=torch.bool,
            device=future_vis.device,
        )

        if self.config.token_dropout_prob > 0.0:
            keep = torch.rand(
                future_vis.shape[:2],
                device=future_vis.device,
                generator=rng,
            ) >= self.config.token_dropout_prob
            noised = noised * keep.unsqueeze(-1)
            future_mask = future_mask & keep

        future_source_id = torch.full(
            (future_vis.shape[0],),
            FUTURE_SOURCE_IDS[self.name],
            dtype=torch.long,
            device=future_vis.device,
        )
        return FutureConditionOutput(
            future_vis=noised,
            future_mask=future_mask,
            future_source_id=future_source_id,
            metadata={"mean_alpha": float(alpha.mean().item())},
        )
