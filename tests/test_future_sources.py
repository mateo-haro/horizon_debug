from __future__ import annotations

import torch

from libero_future_policy.backbones.cosmos_adapter import CosmosAdapter, CosmosAdapterConfig
from libero_future_policy.future_sources.current_only import CurrentOnlyFutureSource
from libero_future_policy.future_sources.mixed_future import MixedFutureSource
from libero_future_policy.future_sources.oracle_clean import OracleCleanFutureSource
from libero_future_policy.future_sources.oracle_noised import OracleNoisedConfig, OracleNoisedFutureSource


def build_batch() -> dict[str, torch.Tensor | list[str] | None]:
    return {
        "current_frames": torch.randn(4, 2, 3, 32, 32),
        "future_frames": torch.randn(4, 3, 3, 32, 32),
        "actions": torch.randn(4, 5, 7),
        "proprio": torch.randn(4, 2, 8),
        "task_text": ["task a", "task b", "task c", "task d"],
    }


def test_future_source_shapes() -> None:
    batch = build_batch()
    backbone = CosmosAdapter(CosmosAdapterConfig(feature_dim=64, image_channels=3))

    current_only = CurrentOnlyFutureSource()
    oracle_clean = OracleCleanFutureSource()
    oracle_noised = OracleNoisedFutureSource(OracleNoisedConfig(noise_std=0.3, token_dropout_prob=0.0))

    current_out = current_only.get_future_condition(batch, backbone, mode="train", step=0)
    clean_out = oracle_clean.get_future_condition(batch, backbone, mode="train", step=0)
    noisy_out = oracle_noised.get_future_condition(batch, backbone, mode="train", step=0)

    assert current_out.future_vis.shape == (4, 3, 64)
    assert current_out.future_mask.shape == (4, 3)
    assert clean_out.future_vis.shape == (4, 3, 64)
    assert noisy_out.future_vis.shape == (4, 3, 64)
    assert not torch.allclose(clean_out.future_vis, noisy_out.future_vis)


def test_mixed_future_source_runs() -> None:
    batch = build_batch()
    backbone = CosmosAdapter(CosmosAdapterConfig(feature_dim=32, image_channels=3))
    source = MixedFutureSource(
        sources={
            "current_only": CurrentOnlyFutureSource(),
            "oracle_clean": OracleCleanFutureSource(),
            "oracle_noised": OracleNoisedFutureSource(OracleNoisedConfig()),
        },
        probabilities={"current_only": 0.2, "oracle_clean": 0.5, "oracle_noised": 0.3},
    )
    out = source.get_future_condition(batch, backbone, mode="train", step=0)
    assert out.future_vis.shape == (4, 3, 32)
    assert out.future_source_id.shape == (4,)
    assert "source_usage" in out.metadata
