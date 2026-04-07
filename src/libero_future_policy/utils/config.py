from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentMetaConfig:
    name: str = "debug"


@dataclass
class DataConfig:
    dataset_path: str | None = None
    mock_dataset: bool = True
    num_mock_trajectories: int = 16
    mock_trajectory_length: int = 48
    image_size: int = 64
    image_channels: int = 3
    current_window: int = 2
    future_window: int = 2
    future_horizon: int = 4
    action_chunk: int = 8
    action_dim: int = 7
    proprio_dim: int = 8
    include_proprio: bool = True
    include_text: bool = True
    index_stride: int = 1


@dataclass
class BackboneConfig:
    feature_dim: int = 128
    use_external_cosmos: bool = False
    external_cosmos_module: str | None = None
    freeze_backbone: bool = False
    cosmos_block_index: int = 0
    cosmos_model_size: str = "2B"
    cosmos_resolution: str = "720"
    cosmos_fps: int = 16
    cosmos_aspect_ratio: str = "16:9"
    cosmos_natten: bool = False
    cosmos_dit_path: str | None = None
    cosmos_checkpoints_root: str | None = None
    cosmos_auto_fetch_checkpoints: bool = True
    cosmos_default_prompt: str = ""
    cosmos_num_conditional_frames: int = 1
    cosmos_intermediate_pool: str = "mean"


@dataclass
class ModelConfig:
    action_dim: int = 7
    action_chunk: int = 8
    proprio_dim: int = 8
    vis_feature_dim: int = 128
    model_dim: int = 256
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    text_vocab_size: int = 4096
    max_text_tokens: int = 16


@dataclass
class TrainConfig:
    seed: int = 0
    device: str = "cpu"
    batch_size: int = 8
    num_workers: int = 0
    max_steps: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    log_every: int = 10
    sample_every: int = 50
    flow_sample_steps: int = 16


@dataclass
class FutureSourceConfig:
    name: str = "oracle_clean"
    noise_std: float = 0.2
    token_dropout_prob: float = 0.1
    mix_probabilities: dict[str, float] = field(
        default_factory=lambda: {
            "current_only": 0.2,
            "oracle_clean": 0.5,
            "oracle_noised": 0.3,
        }
    )


@dataclass
class ExperimentConfig:
    experiment: ExperimentMetaConfig
    data: DataConfig
    backbone: BackboneConfig
    model: ModelConfig
    train: TrainConfig
    future_source: FutureSourceConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_recursive(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    merged: dict[str, Any] = {}
    for include in raw.pop("includes", []):
        include_path = (path.parent / include).resolve()
        merged = _deep_merge(merged, _load_yaml_recursive(include_path))
    return _deep_merge(merged, raw)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    raw = _load_yaml_recursive(config_path)

    data_cfg = DataConfig(**raw.get("data", {}))
    backbone_cfg = BackboneConfig(**raw.get("backbone", {}))
    model_cfg = ModelConfig(**raw.get("model", {}))
    train_cfg = TrainConfig(**raw.get("train", {}))
    future_cfg = FutureSourceConfig(**raw.get("future_source", {}))
    experiment_cfg = ExperimentMetaConfig(**raw.get("experiment", {}))

    if model_cfg.action_dim != data_cfg.action_dim:
        model_cfg.action_dim = data_cfg.action_dim
    if model_cfg.action_chunk != data_cfg.action_chunk:
        model_cfg.action_chunk = data_cfg.action_chunk
    if model_cfg.proprio_dim != data_cfg.proprio_dim:
        model_cfg.proprio_dim = data_cfg.proprio_dim
    if model_cfg.vis_feature_dim != backbone_cfg.feature_dim:
        model_cfg.vis_feature_dim = backbone_cfg.feature_dim

    return ExperimentConfig(
        experiment=experiment_cfg,
        data=data_cfg,
        backbone=backbone_cfg,
        model=model_cfg,
        train=train_cfg,
        future_source=future_cfg,
    )
