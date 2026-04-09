from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class IdentityProcessorPipeline:
    name: str

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        return batch

    def transform_features(self, features: dict[str, Any]) -> dict[str, Any]:
        return features


def make_horizon_dit_pre_post_processors(config) -> tuple[IdentityProcessorPipeline, IdentityProcessorPipeline]:
    """Minimal processor hooks for LeRobot custom-policy integration.

    The policy performs its own batch adaptation internally, so the default
    processors are intentionally identity transforms.
    """

    return IdentityProcessorPipeline("horizon_pre"), IdentityProcessorPipeline("horizon_post")
