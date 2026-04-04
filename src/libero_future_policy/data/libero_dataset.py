from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from libero_future_policy.data.window_sampler import WindowSampler, WindowSpec


@dataclass
class TrajectoryRecord:
    """Single LIBERO-style trajectory.

    Shapes:
    - frames: [T, C, H, W]
    - actions: [T, A]
    - proprio: [T, P] or None
    """

    frames: torch.Tensor
    actions: torch.Tensor
    proprio: torch.Tensor | None
    task_text: str


class LiberoDataset(Dataset[dict[str, Any]]):
    """Windowed LIBERO dataset with a synthetic fallback mode."""

    def __init__(
        self,
        dataset_path: str | None,
        window_spec: WindowSpec,
        mock_dataset: bool = True,
        num_mock_trajectories: int = 16,
        mock_trajectory_length: int = 48,
        image_size: int = 64,
        image_channels: int = 3,
        action_dim: int = 7,
        proprio_dim: int = 8,
        include_proprio: bool = True,
        include_text: bool = True,
        seed: int = 0,
    ) -> None:
        self.window_spec = window_spec
        self.include_text = include_text
        self.include_proprio = include_proprio
        self.generator = torch.Generator().manual_seed(seed)

        if dataset_path is None and mock_dataset:
            self.trajectories = self._build_mock_trajectories(
                num_mock_trajectories=num_mock_trajectories,
                trajectory_length=mock_trajectory_length,
                image_size=image_size,
                image_channels=image_channels,
                action_dim=action_dim,
                proprio_dim=proprio_dim,
                include_proprio=include_proprio,
                include_text=include_text,
            )
        else:
            if dataset_path is None:
                raise ValueError("dataset_path must be set when mock_dataset=False")
            self.trajectories = self._load_trajectories(Path(dataset_path))

        self.sampler = WindowSampler(window_spec)
        self.index: list[tuple[int, int]] = []
        for traj_idx, traj in enumerate(self.trajectories):
            for current_t in self.sampler.build_time_indices(traj.frames.shape[0]):
                self.index.append((traj_idx, current_t))

        if not self.index:
            raise ValueError("No valid training windows were found.")

    def _build_mock_trajectories(
        self,
        num_mock_trajectories: int,
        trajectory_length: int,
        image_size: int,
        image_channels: int,
        action_dim: int,
        proprio_dim: int,
        include_proprio: bool,
        include_text: bool,
    ) -> list[TrajectoryRecord]:
        trajectories: list[TrajectoryRecord] = []
        task_bank = [
            "open the top drawer",
            "pick the cup and place it on the plate",
            "push the block to the goal",
            "stack the red block on the blue block",
        ]

        base_x = torch.linspace(-1.0, 1.0, image_size)
        grid_y, grid_x = torch.meshgrid(base_x, base_x, indexing="ij")
        coord_template = torch.stack([grid_x, grid_y], dim=0)

        for traj_idx in range(num_mock_trajectories):
            frames = []
            actions = []
            proprio = []
            phase = torch.rand(1, generator=self.generator).item() * math.pi
            amplitude = 0.2 + 0.1 * torch.rand(1, generator=self.generator).item()

            for t in range(trajectory_length):
                motion = torch.sin(torch.tensor(t / 4.0 + phase))
                image = torch.zeros(image_channels, image_size, image_size)
                image[0] = coord_template[0]
                image[1] = coord_template[1] * motion
                image[2] = motion + amplitude * coord_template[0] * coord_template[1]
                image = (image + 1.0) * 0.5
                image = image.clamp(0.0, 1.0)
                image = image + 0.01 * torch.randn(
                    image.shape,
                    generator=self.generator,
                    dtype=image.dtype,
                )
                frames.append(image)

                action_t = torch.stack(
                    [
                        torch.sin(torch.tensor(t / 6.0 + phase)),
                        torch.cos(torch.tensor(t / 5.0 + phase)),
                        torch.sin(torch.tensor(t / 7.0)),
                        torch.cos(torch.tensor(t / 8.0)),
                        torch.tensor(amplitude),
                        torch.tensor(float(traj_idx) / max(1, num_mock_trajectories - 1)),
                        torch.tensor(float(t) / trajectory_length),
                    ]
                )[:action_dim]
                actions.append(action_t)

                if include_proprio:
                    prop_t = torch.stack(
                        [
                            torch.sin(torch.tensor(t / 9.0 + phase)),
                            torch.cos(torch.tensor(t / 10.0 + phase)),
                            torch.tensor(float(t) / trajectory_length),
                            torch.tensor(amplitude),
                            motion.float(),
                            torch.tensor(float(traj_idx % 3)),
                            torch.tensor(float(traj_idx % 5)),
                            torch.tensor(1.0),
                        ]
                    )[:proprio_dim]
                    proprio.append(prop_t)

            trajectories.append(
                TrajectoryRecord(
                    frames=torch.stack(frames, dim=0).float(),
                    actions=torch.stack(actions, dim=0).float(),
                    proprio=torch.stack(proprio, dim=0).float() if include_proprio else None,
                    task_text=task_bank[traj_idx % len(task_bank)] if include_text else "",
                )
            )
        return trajectories

    def _load_trajectories(self, dataset_path: Path) -> list[TrajectoryRecord]:
        if dataset_path.suffix in {".pt", ".pth"}:
            raw = torch.load(dataset_path, map_location="cpu")
            return self._coerce_loaded_object(raw)
        if dataset_path.suffix == ".npz":
            import numpy as np

            raw_np = np.load(dataset_path, allow_pickle=True)
            return self._coerce_loaded_object(raw_np["trajectories"].tolist())
        if dataset_path.is_dir():
            records: list[TrajectoryRecord] = []
            for child in sorted(dataset_path.iterdir()):
                if child.suffix in {".pt", ".pth"}:
                    records.extend(self._coerce_loaded_object(torch.load(child, map_location="cpu")))
                elif child.suffix == ".npz":
                    import numpy as np

                    raw_np = np.load(child, allow_pickle=True)
                    records.extend(self._coerce_loaded_object(raw_np["trajectories"].tolist()))
            if records:
                return records
        raise ValueError(
            f"Unsupported dataset format at {dataset_path}. "
            "Expected .pt/.pth/.npz or a directory containing them."
        )

    def _coerce_loaded_object(self, raw: Any) -> list[TrajectoryRecord]:
        if isinstance(raw, dict) and "trajectories" in raw:
            raw = raw["trajectories"]
        if not isinstance(raw, list):
            raise ValueError("Loaded dataset must contain a list of trajectories.")

        records: list[TrajectoryRecord] = []
        for item in raw:
            frames = torch.as_tensor(item["frames"], dtype=torch.float32)
            actions = torch.as_tensor(item["actions"], dtype=torch.float32)
            proprio_raw = item.get("proprio")
            proprio = None if proprio_raw is None else torch.as_tensor(proprio_raw, dtype=torch.float32)
            task_text = str(item.get("task_text", ""))
            records.append(
                TrajectoryRecord(
                    frames=frames,
                    actions=actions,
                    proprio=proprio,
                    task_text=task_text,
                )
            )
        return records

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        traj_idx, current_t = self.index[idx]
        traj = self.trajectories[traj_idx]
        spec = self.window_spec

        curr_start = current_t - spec.current_window + 1
        future_start = current_t + spec.future_horizon
        future_end = future_start + spec.future_window
        action_end = current_t + spec.action_chunk

        current_frames = traj.frames[curr_start : current_t + 1]
        future_frames = traj.frames[future_start:future_end]
        actions = traj.actions[current_t:action_end]

        assert current_frames.shape[0] == spec.current_window
        assert future_frames.shape[0] == spec.future_window
        assert actions.shape[0] == spec.action_chunk

        sample: dict[str, Any] = {
            "current_frames": current_frames,
            "future_frames": future_frames,
            "actions": actions,
            "trajectory_index": torch.tensor(traj_idx, dtype=torch.long),
            "time_index": torch.tensor(current_t, dtype=torch.long),
        }

        if traj.proprio is not None:
            sample["proprio"] = traj.proprio[curr_start : current_t + 1]
        else:
            sample["proprio"] = None

        sample["task_text"] = traj.task_text if self.include_text else ""
        return sample
