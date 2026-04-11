from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from horizon.future_sources.base import FUTURE_SOURCE_IDS, FutureConditionOutput, FutureSource


def _extract_optional_texts(batch: dict[str, Any], batch_size: int) -> list[str] | None:
    for key in ("task_text", "task", "language"):
        if key not in batch:
            continue
        value = batch[key]
        if isinstance(value, list):
            texts = [str(item) for item in value]
        elif isinstance(value, tuple):
            texts = [str(item) for item in value]
        elif isinstance(value, torch.Tensor):
            texts = [str(item) for item in value.tolist()]
        else:
            texts = [str(value) for _ in range(batch_size)]
        if len(texts) == batch_size:
            return texts
    return None


class CurrentOnlyFutureSource(FutureSource):
    name = "current_only"

    def get_future_condition(
        self,
        future_frame_views: list[torch.Tensor],
        batch: dict[str, Any],
        backbone: Any,
        mode: str,
    ) -> FutureConditionOutput:
        if not future_frame_views:
            raise ValueError("Future frame views are required to infer future sequence length.")
        batch_size = future_frame_views[0].shape[0]
        future_window = future_frame_views[0].shape[1]
        feature_window = backbone.feature_num_frames(future_window)
        device = future_frame_views[0].device
        future_vis = backbone.empty_future_features(batch_size, future_window, device)
        future_mask = torch.zeros(batch_size, feature_window, dtype=torch.bool, device=device)
        source_ids = torch.full((batch_size,), FUTURE_SOURCE_IDS[self.name], dtype=torch.long, device=device)
        return FutureConditionOutput(future_vis=future_vis, future_mask=future_mask, future_source_id=source_ids)


class OracleCleanFutureSource(FutureSource):
    name = "oracle_clean"

    def get_future_condition(
        self,
        future_frame_views: list[torch.Tensor],
        batch: dict[str, Any],
        backbone: Any,
        mode: str,
    ) -> FutureConditionOutput:
        future_vis = backbone.encode_multiview(future_frame_views)
        future_mask = torch.ones(future_vis.shape[:2], dtype=torch.bool, device=future_vis.device)
        source_ids = torch.full((future_vis.shape[0],), FUTURE_SOURCE_IDS[self.name], dtype=torch.long, device=future_vis.device)
        return FutureConditionOutput(future_vis=future_vis, future_mask=future_mask, future_source_id=source_ids)


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
        future_frame_views: list[torch.Tensor],
        batch: dict[str, Any],
        backbone: Any,
        mode: str,
    ) -> FutureConditionOutput:
        future_vis = backbone.encode_multiview(future_frame_views)
        alpha = torch.rand(future_vis.shape[0], 1, 1, device=future_vis.device)
        noised = future_vis + alpha * self.config.noise_std * torch.randn_like(future_vis)
        future_mask = torch.ones(future_vis.shape[:2], dtype=torch.bool, device=future_vis.device)
        if self.config.token_dropout_prob > 0.0 and mode == "train":
            keep = torch.rand(future_vis.shape[:2], device=future_vis.device) >= self.config.token_dropout_prob
            noised = noised * keep.unsqueeze(-1)
            future_mask = future_mask & keep
        source_ids = torch.full((future_vis.shape[0],), FUTURE_SOURCE_IDS[self.name], dtype=torch.long, device=future_vis.device)
        return FutureConditionOutput(
            future_vis=noised,
            future_mask=future_mask,
            future_source_id=source_ids,
            metadata={"mean_alpha": float(alpha.mean().item())},
        )


class OracleHiddenFutureSource(FutureSource):
    name = "oracle_hidden"

    def __init__(self, tau: float, hidden_layer_index: int, noise_level: float) -> None:
        self.tau = tau
        self.hidden_layer_index = hidden_layer_index
        self.noise_level = noise_level

    def get_future_condition(
        self,
        future_frame_views: list[torch.Tensor],
        batch: dict[str, Any],
        backbone: Any,
        mode: str,
    ) -> FutureConditionOutput:
        texts = _extract_optional_texts(batch, future_frame_views[0].shape[0])
        future_vis = backbone.extract_hidden_features_multiview(
            frame_views=future_frame_views,
            tau=self.tau,
            hidden_layer_index=self.hidden_layer_index,
            noise_level=self.noise_level,
            texts=texts,
        )
        future_mask = torch.ones(future_vis.shape[:2], dtype=torch.bool, device=future_vis.device)
        source_ids = torch.full((future_vis.shape[0],), FUTURE_SOURCE_IDS[self.name], dtype=torch.long, device=future_vis.device)
        return FutureConditionOutput(
            future_vis=future_vis,
            future_mask=future_mask,
            future_source_id=source_ids,
            metadata={"tau": self.tau, "hidden_layer": self.hidden_layer_index},
        )


class GeneratedFutureSource(FutureSource):
    name = "generated"

    def get_future_condition(
        self,
        future_frame_views: list[torch.Tensor],
        batch: dict[str, Any],
        backbone: Any,
        mode: str,
    ) -> FutureConditionOutput:
        raise NotImplementedError(
            "Generated future conditioning requires a real frozen Cosmos rollout path and is still a TODO."
        )


class MixedFutureSource(FutureSource):
    name = "mixed"

    def __init__(self, sources: dict[str, FutureSource], probabilities: dict[str, float]) -> None:
        self.sources = sources
        self.names = list(sources.keys())
        probs = torch.tensor([probabilities[name] for name in self.names], dtype=torch.float32)
        self.probabilities = probs / probs.sum()

    def get_future_condition(
        self,
        future_frame_views: list[torch.Tensor],
        batch: dict[str, Any],
        backbone: Any,
        mode: str,
    ) -> FutureConditionOutput:
        batch_size = future_frame_views[0].shape[0]
        device = future_frame_views[0].device
        future_window = future_frame_views[0].shape[1]
        feature_window = backbone.feature_num_frames(future_window)

        choices = torch.multinomial(self.probabilities, num_samples=batch_size, replacement=True)
        future_vis = backbone.empty_future_features(batch_size, future_window, device)
        future_mask = torch.zeros(batch_size, feature_window, dtype=torch.bool, device=device)
        future_source_id = torch.zeros(batch_size, dtype=torch.long, device=device)

        usage: dict[str, int] = {}
        for source_index, name in enumerate(self.names):
            indices = (choices == source_index).nonzero(as_tuple=False).flatten()
            if indices.numel() == 0:
                continue

            usage[name] = int(indices.numel())
            sub_views = [view.index_select(0, indices) for view in future_frame_views]
            sub_output = self.sources[name].get_future_condition(sub_views, batch, backbone, mode)
            future_vis.index_copy_(0, indices, sub_output.future_vis)
            future_mask.index_copy_(0, indices, sub_output.future_mask)
            future_source_id.index_copy_(0, indices, sub_output.future_source_id)

        return FutureConditionOutput(
            future_vis=future_vis,
            future_mask=future_mask,
            future_source_id=future_source_id,
            metadata={"source_usage": usage},
        )
