from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class HorizonLaunchConfig:
    train: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)


def load_horizon_launch_config(path: str | Path) -> HorizonLaunchConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return HorizonLaunchConfig(
        train=dict(raw.get("train", {})),
        policy=dict(raw.get("policy", {})),
    )
