from __future__ import annotations

import torch

from horizon.configuration_horizon_dit import HorizonDiTConfig
from horizon.modeling_horizon_dit import HorizonDiTPolicy


def make_batch(batch_size: int = 2, total_obs_steps: int = 4, action_chunk: int = 8) -> dict[str, torch.Tensor | list[str]]:
    return {
        "observation.images.image": torch.randn(batch_size, total_obs_steps, 3, 64, 64),
        "observation.state": torch.randn(batch_size, total_obs_steps, 8),
        "action": torch.randn(batch_size, action_chunk, 7),
        "task_text": ["open the drawer", "pick the mug"],
    }


def test_policy_forward_runs() -> None:
    config = HorizonDiTConfig(
        device="cpu",
        current_obs_steps=2,
        future_obs_steps=2,
        chunk_size=8,
        n_action_steps=4,
        model_dim=128,
        depth=2,
        num_heads=4,
        vis_feature_dim=64,
        cosmos_feature_dim=64,
        cosmos_num_hidden_layers=3,
        cosmos_hidden_layer=2,
        future_source="oracle_hidden",
        use_task_text=True,
        use_proprio=True,
        libero_suite=None,
    )
    policy = HorizonDiTPolicy(config)
    batch = make_batch()
    loss, info = policy.forward(batch)
    assert loss.ndim == 0
    assert "future_mask_frac" in info


def test_policy_select_action_runs() -> None:
    config = HorizonDiTConfig(
        device="cpu",
        current_obs_steps=2,
        future_obs_steps=2,
        chunk_size=6,
        n_action_steps=3,
        model_dim=128,
        depth=2,
        num_heads=4,
        vis_feature_dim=64,
        cosmos_feature_dim=64,
        cosmos_num_hidden_layers=3,
        cosmos_hidden_layer=2,
        future_source="mixed",
        use_task_text=False,
        use_proprio=True,
        libero_suite="libero_spatial",
    )
    policy = HorizonDiTPolicy(config)
    batch = make_batch(action_chunk=6)
    action0 = policy.select_action(batch)
    action1 = policy.select_action(batch)
    assert action0.shape == (2, 7)
    assert action1.shape == (2, 7)
