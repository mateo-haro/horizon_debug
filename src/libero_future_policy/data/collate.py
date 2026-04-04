from __future__ import annotations

from typing import Any

import torch


def libero_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    current_frames = torch.stack([item["current_frames"] for item in batch], dim=0)
    future_frames = torch.stack([item["future_frames"] for item in batch], dim=0)
    actions = torch.stack([item["actions"] for item in batch], dim=0)
    trajectory_index = torch.stack([item["trajectory_index"] for item in batch], dim=0)
    time_index = torch.stack([item["time_index"] for item in batch], dim=0)

    proprio_items = [item["proprio"] for item in batch]
    proprio = None if any(x is None for x in proprio_items) else torch.stack(proprio_items, dim=0)

    return {
        "current_frames": current_frames,
        "future_frames": future_frames,
        "actions": actions,
        "proprio": proprio,
        "task_text": [str(item["task_text"]) for item in batch],
        "trajectory_index": trajectory_index,
        "time_index": time_index,
    }
