from __future__ import annotations

from typing import Any

import torch

from horizon.libero import task_texts_from_indices


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _ensure_sequence_tensor(value: torch.Tensor, is_image: bool) -> torch.Tensor:
    if is_image:
        if value.ndim == 4:
            return value.unsqueeze(1)
        if value.ndim != 5:
            raise ValueError(f"Expected image tensor rank 4 or 5, got shape {tuple(value.shape)}")
        return value
    if value.ndim == 2:
        return value.unsqueeze(1)
    if value.ndim != 3:
        raise ValueError(f"Expected sequence tensor rank 2 or 3, got shape {tuple(value.shape)}")
    return value


def get_image_keys(batch: dict[str, Any], preferred_keys: list[str]) -> list[str]:
    present = [key for key in preferred_keys if key in batch]
    if present:
        return present
    discovered = sorted(
        key
        for key in batch.keys()
        if key.startswith("observation.images") or key.startswith("observation.image")
    )
    if not discovered:
        raise KeyError("No image observation keys were found in the batch.")
    return discovered


def get_image_sequences(batch: dict[str, Any], preferred_keys: list[str]) -> list[torch.Tensor]:
    keys = get_image_keys(batch, preferred_keys)
    return [_ensure_sequence_tensor(batch[key], is_image=True) for key in keys]


def get_state_sequence(batch: dict[str, Any], state_key: str | None) -> torch.Tensor | None:
    if state_key is None or state_key not in batch:
        return None
    return _ensure_sequence_tensor(batch[state_key], is_image=False)


def split_current_future(
    sequence: torch.Tensor,
    current_obs_steps: int,
    future_obs_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    current = sequence[:, :current_obs_steps]
    future = sequence[:, current_obs_steps : current_obs_steps + future_obs_steps]
    return current, future


def get_action_chunk(batch: dict[str, Any], chunk_size: int) -> torch.Tensor:
    if "action" not in batch:
        raise KeyError("Expected `action` in batch.")
    actions = batch["action"]
    if actions.ndim == 2:
        actions = actions.unsqueeze(1)
    if actions.ndim != 3:
        raise ValueError(f"Expected `action` rank 2 or 3, got shape {tuple(actions.shape)}")
    if actions.shape[1] < chunk_size:
        raise ValueError(f"Need action chunk length {chunk_size}, got shape {tuple(actions.shape)}")
    return actions[:, :chunk_size]


def _extract_task_indices(batch: dict[str, Any]) -> list[int] | None:
    for key in ["task_index", "task_idx", "episode.task_index", "observation.task_index"]:
        if key not in batch:
            continue
        value = batch[key]
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return [int(value.item())]
            if value.ndim == 1:
                return [int(x) for x in value.tolist()]
            if value.ndim >= 2:
                return [int(x) for x in value[:, 0].tolist()]
    return None


def get_task_texts(batch: dict[str, Any], suite: str | None) -> list[str]:
    batch_size = infer_batch_size(batch)
    for key in ["task", "task_text", "language"]:
        if key not in batch:
            continue
        value = batch[key]
        if isinstance(value, list):
            return [str(x) for x in value]
        if isinstance(value, tuple):
            return [str(x) for x in value]
        if isinstance(value, torch.Tensor):
            return [str(x) for x in value.tolist()]
        return [str(value) for _ in range(batch_size)]

    if suite is not None:
        indices = _extract_task_indices(batch)
        if indices is not None:
            return task_texts_from_indices(suite, indices)

    return ["" for _ in range(batch_size)]


def infer_batch_size(batch: dict[str, Any]) -> int:
    for value in batch.values():
        if isinstance(value, torch.Tensor):
            return int(value.shape[0])
        if isinstance(value, list):
            return len(value)
    raise ValueError("Unable to infer batch size from batch.")
