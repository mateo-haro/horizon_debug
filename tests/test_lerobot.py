from __future__ import annotations

from utils.config import HorizonLaunchConfig
from utils.lerobot import build_train_command


def test_build_train_command_defaults() -> None:
    config = HorizonLaunchConfig(
        train={
            "suite": "libero_spatial",
            "dataset_flavor": "image",
            "output_dir": "outputs/train/libero_spatial_horizon_dit",
            "job_name": "libero_spatial_horizon_dit",
            "train_entrypoint": "lerobot-train",
            "wandb_enable": False,
            "extra_args": [],
        },
        policy={"type": "horizon_dit", "device": "cuda", "future_source": "oracle_hidden"},
    )
    command = build_train_command(config)
    assert command[0] == "lerobot-train"
    assert "--dataset.repo_id=lerobot/libero_spatial_image" in command
    assert "--policy.type=horizon_dit" in command
    assert "--policy.libero_suite=libero_spatial" in command
    assert "--wandb.enable=false" in command


def test_build_train_command_nested_policy_args() -> None:
    config = HorizonLaunchConfig(
        train={
            "suite": "libero_goal",
            "dataset_flavor": "image",
            "output_dir": "outputs/train/libero_goal_horizon_dit",
            "job_name": "libero_goal_horizon_dit",
            "train_entrypoint": "lerobot-train",
            "wandb_enable": True,
            "extra_args": [],
        },
        policy={
            "type": "horizon_dit",
            "device": "cuda",
            "future_source": "mixed",
            "source_mix": {"oracle_hidden": 0.4, "oracle_clean": 0.6},
        },
    )
    command = build_train_command(config)
    assert "--policy.source_mix.oracle_hidden=0.4" in command
    assert "--policy.source_mix.oracle_clean=0.6" in command
    assert "--wandb.enable=true" in command


def test_passthrough_overrides_base_flag() -> None:
    config = HorizonLaunchConfig(
        train={
            "suite": "libero_spatial",
            "dataset_flavor": "image",
            "output_dir": "outputs/train/libero_spatial_horizon_dit",
            "job_name": "libero_spatial_horizon_dit",
            "train_entrypoint": "lerobot-train",
            "wandb_enable": False,
            "extra_args": [],
        },
        policy={"type": "horizon_dit"},
    )
    command = build_train_command(config, ["--wandb.enable=true"])
    assert "--wandb.enable=false" not in command
    assert "--wandb.enable=true" in command
