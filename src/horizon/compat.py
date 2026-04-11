from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from torch import Tensor, nn

try:
    from lerobot.common.optim.optimizers import AdamWConfig  # type: ignore
    from lerobot.common.optim.schedulers import LRSchedulerConfig  # type: ignore
    from lerobot.common.policies.pretrained import PreTrainedPolicy  # type: ignore
    from lerobot.configs.policies import PreTrainedConfig  # type: ignore
    from lerobot.configs.types import FeatureType, PolicyFeature  # type: ignore

    LEROBOT_AVAILABLE = True
except Exception:  # pragma: no cover - exercised when lerobot is not installed
    LEROBOT_AVAILABLE = False

    class FeatureType(str, Enum):
        VISUAL = "visual"
        STATE = "state"
        ACTION = "action"
        TEXT = "text"

    @dataclass
    class PolicyFeature:
        type: FeatureType
        shape: tuple[int, ...]

    @dataclass
    class PreTrainedConfig:
        input_features: dict[str, PolicyFeature] = field(default_factory=dict)
        output_features: dict[str, PolicyFeature] = field(default_factory=dict)
        device: str = "cpu"

        @classmethod
        def register_subclass(cls, _name: str):
            def decorator(subclass):
                return subclass

            return decorator

        @classmethod
        def from_pretrained(cls, pretrained_name_or_path: str | Path, **_: Any):
            config_path = Path(pretrained_name_or_path) / "config.yaml"
            with config_path.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
            return cls(**raw)

        def _save_pretrained(self, save_directory: str | Path) -> None:
            save_dir = Path(save_directory)
            save_dir.mkdir(parents=True, exist_ok=True)
            with (save_dir / "config.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(self.__dict__, handle)

    class PreTrainedPolicy(nn.Module):
        config_class = None
        name = None

        def __init__(self, config: PreTrainedConfig, *args, **kwargs):
            super().__init__()
            self.config = config

        def get_optim_params(self) -> dict:
            raise NotImplementedError

        def reset(self):
            raise NotImplementedError

        def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
            raise NotImplementedError

        def select_action(self, batch: dict[str, Tensor]) -> Tensor:
            raise NotImplementedError

    @dataclass
    class AdamWConfig:
        lr: float = 3e-4
        betas: tuple[float, float] = (0.9, 0.999)
        eps: float = 1e-8
        weight_decay: float = 1e-4
        grad_clip_norm: float = 10.0

    class LRSchedulerConfig:  # pragma: no cover - simple placeholder
        pass
