from __future__ import annotations

from dataclasses import dataclass, field

from lerobot_policy_horizon.compat import AdamWConfig, FeatureType, LRSchedulerConfig, PolicyFeature, PreTrainedConfig


def _default_input_features() -> dict[str, PolicyFeature]:
    return {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }


def _default_output_features() -> dict[str, PolicyFeature]:
    return {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }


@PreTrainedConfig.register_subclass("horizon_dit")
@dataclass
class HorizonDiTConfig(PreTrainedConfig):
    input_features: dict[str, PolicyFeature] = field(default_factory=_default_input_features)
    output_features: dict[str, PolicyFeature] = field(default_factory=_default_output_features)
    device: str = "cuda"

    current_obs_steps: int = 2
    future_obs_steps: int = 2
    future_offset: int = 4
    chunk_size: int = 8
    n_action_steps: int = 8

    future_source: str = "oracle_hidden"
    source_mix: dict[str, float] = field(
        default_factory=lambda: {
            "current_only": 0.2,
            "oracle_clean": 0.3,
            "oracle_noised": 0.2,
            "oracle_hidden": 0.3,
        }
    )

    vis_feature_dim: int = 256
    model_dim: int = 512
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    text_vocab_size: int = 4096
    max_text_tokens: int = 16
    flow_matching_steps: int = 16

    cosmos_feature_dim: int = 256
    cosmos_num_hidden_layers: int = 8
    cosmos_hidden_layer: int = 3
    cosmos_noise_level: float = 0.15
    cosmos_tau: float = 0.5
    cosmos_use_external_runtime: bool = False
    cosmos_external_module: str | None = None
    cosmos_repo_path: str | None = "cosmos-predict2"
    cosmos_model_size: str = "2B"
    cosmos_resolution: str = "480"
    cosmos_fps: int = 10
    cosmos_lora_checkpoint: str | None = None
    cosmos_lora_rank: int = 16
    cosmos_lora_alpha: int = 16
    cosmos_lora_target_modules: str = "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2"
    cosmos_prompt_refiner_enabled: bool = False
    cosmos_guardrail_enabled: bool = False

    use_proprio: bool = True
    use_task_text: bool = True
    libero_suite: str | None = None

    optimizer_lr: float = 3e-4
    optimizer_weight_decay: float = 1e-4

    @property
    def observation_delta_indices(self) -> list[int]:
        current = list(range(1 - self.current_obs_steps, 1))
        future = list(range(self.future_offset, self.future_offset + self.future_obs_steps))
        return current + future

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> list[int] | None:
        return None

    @property
    def image_keys(self) -> list[str]:
        return [key for key, value in self.input_features.items() if value.type == FeatureType.VISUAL]

    @property
    def proprio_key(self) -> str | None:
        for key, value in self.input_features.items():
            if value.type == FeatureType.STATE:
                return key
        return None

    @property
    def proprio_dim(self) -> int:
        key = self.proprio_key
        if key is None:
            return 0
        return int(self.input_features[key].shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.output_features["action"].shape[0])

    def validate_features(self) -> None:
        if "action" not in self.output_features:
            raise ValueError("`action` must be present in output_features.")
        if not self.image_keys:
            raise ValueError("At least one visual observation feature is required.")

    def get_optimizer_preset(self):
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None
