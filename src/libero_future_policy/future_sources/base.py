from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


FUTURE_SOURCE_IDS = {
    "current_only": 0,
    "oracle_clean": 1,
    "oracle_noised": 2,
    "generated": 3,
}


@dataclass
class FutureConditionOutput:
    future_vis: torch.Tensor
    future_mask: torch.Tensor
    future_source_id: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


class FutureSource(ABC):
    name: str

    @abstractmethod
    def get_future_condition(
        self,
        batch: dict[str, Any],
        backbone: Any,
        mode: str,
        step: int,
        rng: torch.Generator | None = None,
    ) -> FutureConditionOutput:
        raise NotImplementedError


def index_batch(batch: dict[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    index_list = indices.tolist()
    for key, value in batch.items():
        if value is None:
            sliced[key] = None
        elif isinstance(value, torch.Tensor):
            sliced[key] = value.index_select(0, indices)
        elif isinstance(value, list):
            sliced[key] = [value[i] for i in index_list]
        else:
            sliced[key] = value
    return sliced
