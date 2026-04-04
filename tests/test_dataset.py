from __future__ import annotations

from libero_future_policy.data.libero_dataset import LiberoDataset
from libero_future_policy.data.window_sampler import WindowSpec


def test_dataset_window_shapes() -> None:
    dataset = LiberoDataset(
        dataset_path=None,
        window_spec=WindowSpec(current_window=2, future_window=3, future_horizon=4, action_chunk=5),
        mock_dataset=True,
        num_mock_trajectories=2,
        mock_trajectory_length=24,
        image_size=32,
        action_dim=7,
        proprio_dim=8,
        include_proprio=True,
        include_text=True,
    )

    sample = dataset[0]
    assert sample["current_frames"].shape == (2, 3, 32, 32)
    assert sample["future_frames"].shape == (3, 3, 32, 32)
    assert sample["actions"].shape == (5, 7)
    assert sample["proprio"].shape == (2, 8)
    assert isinstance(sample["task_text"], str)
