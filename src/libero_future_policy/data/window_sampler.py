from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowSpec:
    current_window: int
    future_window: int
    future_horizon: int
    action_chunk: int
    index_stride: int = 1


class WindowSampler:
    """Build valid current-time indices for a trajectory."""

    def __init__(self, spec: WindowSpec) -> None:
        self.spec = spec

    def min_current_t(self) -> int:
        return self.spec.current_window - 1

    def max_current_t(self, trajectory_length: int) -> int:
        future_limit = trajectory_length - (self.spec.future_horizon + self.spec.future_window)
        action_limit = trajectory_length - self.spec.action_chunk
        return min(future_limit, action_limit)

    def build_time_indices(self, trajectory_length: int) -> list[int]:
        max_t = self.max_current_t(trajectory_length)
        min_t = self.min_current_t()
        if max_t < min_t:
            return []
        return list(range(min_t, max_t + 1, self.spec.index_stride))
