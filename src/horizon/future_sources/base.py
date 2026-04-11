from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


FUTURE_SOURCE_IDS = {
    "current_only": 0,
    "oracle_clean": 1,
    "oracle_noised": 2,
    "oracle_hidden": 3,
    "generated": 4,
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
        future_frame_views: list[torch.Tensor],
        batch: dict[str, Any],
        backbone: Any,
        mode: str,
    ) -> FutureConditionOutput:
        raise NotImplementedError
