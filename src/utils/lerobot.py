from __future__ import annotations

from typing import Any

from utils.config import HorizonLaunchConfig
from utils.libero import dataset_repo_id_for_suite


def _extract_override_keys(args: list[str]) -> set[str]:
    keys: set[str] = set()
    for arg in args:
        if not arg.startswith("--"):
            continue
        stripped = arg[2:]
        if "=" in stripped:
            stripped = stripped.split("=", 1)[0]
        keys.add(stripped)
    return keys


def _flatten_cli_args(prefix: str, value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        flattened: dict[str, str] = {}
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_cli_args(nested_prefix, nested_value))
        return flattened
    if isinstance(value, bool):
        return {prefix: "true" if value else "false"}
    if value is None:
        return {}
    return {prefix: str(value)}


def build_train_command(config: HorizonLaunchConfig, passthrough_args: list[str] | None = None) -> list[str]:
    train = dict(config.train)
    policy = dict(config.policy)
    policy.setdefault("libero_suite", train.get("suite", "libero_spatial"))

    dataset_repo_id = train.get("dataset_repo_id") or dataset_repo_id_for_suite(
        suite=train.get("suite", "libero_spatial"),
        dataset_flavor=train.get("dataset_flavor", "image"),
    )

    train_entrypoint = str(train.get("train_entrypoint", "lerobot-train"))
    passthrough_args = passthrough_args or []
    extra_args = list(train.get("extra_args", []))
    override_keys = _extract_override_keys(extra_args + passthrough_args)

    command = [train_entrypoint]
    base_args = {
        "dataset.repo_id": dataset_repo_id,
        "output_dir": train.get("output_dir", "outputs/train/horizon"),
        "job_name": train.get("job_name", "horizon"),
        "wandb.enable": train.get("wandb_enable", False),
    }
    base_args.update(_flatten_cli_args("policy", policy))

    for key, value in base_args.items():
        if key not in override_keys:
            if isinstance(value, bool):
                value = "true" if value else "false"
            command.append(f"--{key}={value}")

    command.extend(extra_args)
    command.extend(passthrough_args)
    return command
