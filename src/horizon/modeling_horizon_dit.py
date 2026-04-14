from __future__ import annotations

from typing import Any

import torch

from horizon.backbones.cosmos_adapter import CosmosAdapter
from horizon.compat import PreTrainedPolicy
from horizon.configuration_horizon_dit import HorizonDiTConfig
from horizon.future_sources.base import FUTURE_SOURCE_IDS
from horizon.future_sources.sources import (
    CurrentOnlyFutureSource,
    GeneratedFutureSource,
    MixedFutureSource,
    OracleCleanFutureSource,
    OracleHiddenFutureSource,
    OracleNoisedConfig,
    OracleNoisedFutureSource,
    PrecomputedFutureSource,
)
from horizon.models.action_dit import ActionDiT, ActionDiTConfig
from horizon.models.action_flow import ActionFlowMatcher
from horizon.models.visual_latent_projection import VisualLatentProjection
from horizon.utils.batch import (
    get_action_chunk,
    get_image_sequences,
    get_state_sequence,
    get_task_texts,
    move_batch_to_device,
    split_current_future,
)


class HorizonDiTPolicy(PreTrainedPolicy):
    config_class = HorizonDiTConfig
    name = "horizon_dit"

    def __init__(self, config: HorizonDiTConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        config.validate_features()

        self.visual_latent_proj = VisualLatentProjection(
            encode_in_dim=config.cosmos_encode_token_dim,
            hidden_in_dim=config.cosmos_hidden_token_dim,
            out_dim=config.cosmos_feature_dim,
            use_latent_projection=config.use_latent_projection,
        )

        if config.use_precomputed_cosmos_latents:
            self.cosmos = None
        else:
            self.cosmos = CosmosAdapter(
                feature_dim=config.cosmos_feature_dim,
                hidden_layer_index=config.cosmos_hidden_layer,
                noise_level=config.cosmos_noise_level,
                num_hidden_layers=config.cosmos_num_hidden_layers,
                device=config.device,
                use_external_runtime=config.cosmos_use_external_runtime,
                external_module=config.cosmos_external_module,
                repo_path=config.cosmos_repo_path,
                model_size=config.cosmos_model_size,
                resolution=config.cosmos_resolution,
                fps=config.cosmos_fps,
                lora_checkpoint=config.cosmos_lora_checkpoint,
                lora_rank=config.cosmos_lora_rank,
                lora_alpha=config.cosmos_lora_alpha,
                lora_target_modules=config.cosmos_lora_target_modules,
                prompt_refiner_enabled=config.cosmos_prompt_refiner_enabled,
                guardrail_enabled=config.cosmos_guardrail_enabled,
            )
        self.model = ActionDiT(
            ActionDiTConfig(
                action_dim=config.action_dim,
                action_chunk=config.chunk_size,
                proprio_dim=max(1, config.proprio_dim),
                vis_feature_dim=config.cosmos_feature_dim,
                model_dim=config.model_dim,
                depth=config.depth,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                text_vocab_size=config.text_vocab_size,
                max_text_tokens=config.max_text_tokens,
                use_proprio=config.use_proprio,
                use_task_text=config.use_task_text,
            )
        )
        self.flow_matcher = ActionFlowMatcher()
        self.future_source = self._build_future_source(config)
        self._cached_actions: torch.Tensor | None = None
        self._cache_index = 0

    def _build_future_source(self, config: HorizonDiTConfig):
        current_only = CurrentOnlyFutureSource()
        oracle_clean = OracleCleanFutureSource()
        oracle_noised = OracleNoisedFutureSource(
            OracleNoisedConfig(
                noise_std=config.cosmos_noise_level,
                token_dropout_prob=min(0.5, config.cosmos_noise_level),
            )
        )
        oracle_hidden = OracleHiddenFutureSource(
            tau=config.cosmos_tau,
            hidden_layer_index=config.cosmos_hidden_layer,
            noise_level=config.cosmos_noise_level,
        )

        if config.future_source == "current_only":
            return current_only
        if config.future_source == "oracle_clean":
            return oracle_clean
        if config.future_source == "oracle_noised":
            return oracle_noised
        if config.future_source == "oracle_hidden":
            return oracle_hidden
        if config.future_source == "precomputed":
            return PrecomputedFutureSource(future_key=config.precomputed_future_key)
        if config.future_source == "generated":
            return GeneratedFutureSource()
        if config.future_source == "mixed":
            return MixedFutureSource(
                sources={
                    "current_only": current_only,
                    "oracle_clean": oracle_clean,
                    "oracle_noised": oracle_noised,
                    "oracle_hidden": oracle_hidden,
                },
                probabilities=config.source_mix,
            )
        raise ValueError(f"Unknown future source: {config.future_source}")

    def _build_conditioning(self, batch: dict[str, Any], mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.config.use_precomputed_cosmos_latents:
            if self.config.precomputed_curr_key not in batch or self.config.precomputed_future_key not in batch:
                raise KeyError(
                    "use_precomputed_cosmos_latents=True requires batch keys "
                    f"{self.config.precomputed_curr_key!r} and {self.config.precomputed_future_key!r}."
                )
            curr_vis = self.visual_latent_proj.forward_encode(batch[self.config.precomputed_curr_key])
            future_vis = self.visual_latent_proj.forward_hidden(batch[self.config.precomputed_future_key])
            future_mask = torch.ones(future_vis.shape[:2], dtype=torch.bool, device=future_vis.device)
            future_source_id = torch.full(
                (curr_vis.shape[0],),
                FUTURE_SOURCE_IDS["precomputed"],
                dtype=torch.long,
                device=curr_vis.device,
            )
            metadata: dict[str, Any] = {}
        else:
            assert self.cosmos is not None
            image_sequences = get_image_sequences(batch, self.config.image_keys)
            current_views = []
            future_views = []
            for sequence in image_sequences:
                current, future = split_current_future(
                    sequence=sequence,
                    current_obs_steps=self.config.current_obs_steps,
                    future_obs_steps=self.config.future_obs_steps,
                )
                current_views.append(current)
                future_views.append(future)

            curr_vis = self.visual_latent_proj.forward_encode(self.cosmos.encode_multiview(current_views))
            future_output = self.future_source.get_future_condition(
                future_frame_views=future_views,
                batch=batch,
                backbone=self.cosmos,
                mode=mode,
            )
            future_vis = self.visual_latent_proj.forward_hidden(future_output.future_vis)
            future_mask = future_output.future_mask
            future_source_id = future_output.future_source_id
            metadata = future_output.metadata

        proprio_sequence = get_state_sequence(batch, self.config.proprio_key)
        proprio = None
        if proprio_sequence is not None and self.config.use_proprio:
            proprio, _ = split_current_future(
                sequence=proprio_sequence,
                current_obs_steps=self.config.current_obs_steps,
                future_obs_steps=self.config.future_obs_steps,
            )

        task_texts = get_task_texts(batch, self.config.libero_suite if self.config.use_task_text else None)
        conditioning = {
            "curr_vis": curr_vis,
            "future_vis": future_vis,
            "future_mask": future_mask,
            "proprio": proprio,
            "text": task_texts,
            "future_source_id": future_source_id,
        }
        return conditioning, metadata

    def get_optim_params(self) -> dict:
        return {"params": self.parameters()}

    def reset(self):
        self._cached_actions = None
        self._cache_index = 0

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        device = next(self.parameters()).device
        batch = move_batch_to_device(batch, device)
        conditioning, metadata = self._build_conditioning(batch, mode="train")
        actions = get_action_chunk(batch, self.config.chunk_size)
        output = self.flow_matcher.training_loss(self.model, actions, conditioning)

        info: dict[str, Any] = {
            "loss": float(output.loss.detach().item()),
            "t_mean": float(output.timesteps.mean().item()),
            "future_mask_frac": float(conditioning["future_mask"].float().mean().item()),
        }
        source_usage = metadata.get("source_usage")
        if source_usage:
            for key, value in source_usage.items():
                info[f"source_{key}"] = int(value)
        if "tau" in metadata:
            info["tau"] = float(metadata["tau"])
        return output.loss, info

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        batch = move_batch_to_device(batch, device)

        if self._cached_actions is not None and self._cache_index < self._cached_actions.shape[1]:
            action = self._cached_actions[:, self._cache_index]
            self._cache_index += 1
            return action

        conditioning, _ = self._build_conditioning(batch, mode="eval")
        curr_vis = conditioning["curr_vis"]
        sampled_chunk = self.flow_matcher.sample(
            model=self.model,
            conditioning=conditioning,
            batch_size=curr_vis.shape[0],
            action_chunk=self.config.chunk_size,
            action_dim=self.config.action_dim,
            device=curr_vis.device,
            num_steps=self.config.flow_matching_steps,
        )
        cache_steps = min(self.config.n_action_steps, sampled_chunk.shape[1])
        self._cached_actions = sampled_chunk[:, :cache_steps]
        self._cache_index = 1
        return self._cached_actions[:, 0]
