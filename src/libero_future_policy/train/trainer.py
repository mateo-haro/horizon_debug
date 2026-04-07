from __future__ import annotations

import itertools
import random
from dataclasses import asdict
from typing import Any, Iterator

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from libero_future_policy.backbones.cosmos_adapter import CosmosAdapter, CosmosAdapterConfig
from libero_future_policy.data.collate import libero_collate
from libero_future_policy.data.libero_dataset import LiberoDataset
from libero_future_policy.data.window_sampler import WindowSpec
from libero_future_policy.future_sources.current_only import CurrentOnlyFutureSource
from libero_future_policy.future_sources.generated_future import GeneratedFutureSource
from libero_future_policy.future_sources.mixed_future import MixedFutureSource
from libero_future_policy.future_sources.oracle_clean import OracleCleanFutureSource
from libero_future_policy.future_sources.oracle_noised import OracleNoisedConfig, OracleNoisedFutureSource
from libero_future_policy.models.action_diffusion import ActionFlowMatcher
from libero_future_policy.models.action_dit import ActionDiT, ActionDiTConfig
from libero_future_policy.train.eval import sample_action_chunk
from libero_future_policy.train.losses import compute_action_flow_matching_loss
from libero_future_policy.utils.config import ExperimentConfig
from libero_future_policy.utils.logging import StepLogger


class Trainer:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.device = torch.device(config.train.device)
        self.logger = StepLogger(config.experiment.name)
        self.rng = torch.Generator().manual_seed(config.train.seed)
        self._seed_everything(config.train.seed)

        self.dataset = self._build_dataset()
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=config.train.num_workers,
            collate_fn=libero_collate,
        )

        self.backbone = self._build_backbone().to(self.device)
        self.model = self._build_model().to(self.device)
        self.flow_matcher = ActionFlowMatcher()
        self.future_source = self._build_future_source()

        parameters = itertools.chain(self.backbone.parameters(), self.model.parameters())
        self.optimizer = AdamW(
            parameters,
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )

        self.step = 0

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)

    def _build_dataset(self) -> LiberoDataset:
        data_cfg = self.config.data
        window_spec = WindowSpec(
            current_window=data_cfg.current_window,
            future_window=data_cfg.future_window,
            future_horizon=data_cfg.future_horizon,
            action_chunk=data_cfg.action_chunk,
            index_stride=data_cfg.index_stride,
        )
        return LiberoDataset(
            dataset_path=data_cfg.dataset_path,
            window_spec=window_spec,
            mock_dataset=data_cfg.mock_dataset,
            num_mock_trajectories=data_cfg.num_mock_trajectories,
            mock_trajectory_length=data_cfg.mock_trajectory_length,
            image_size=data_cfg.image_size,
            image_channels=data_cfg.image_channels,
            action_dim=data_cfg.action_dim,
            proprio_dim=data_cfg.proprio_dim,
            include_proprio=data_cfg.include_proprio,
            include_text=data_cfg.include_text,
            seed=self.config.train.seed,
        )

    def _build_backbone(self) -> CosmosAdapter:
        backbone_cfg = self.config.backbone
        return CosmosAdapter(
            CosmosAdapterConfig(
                feature_dim=backbone_cfg.feature_dim,
                use_external_cosmos=backbone_cfg.use_external_cosmos,
                external_cosmos_module=backbone_cfg.external_cosmos_module,
                freeze_backbone=backbone_cfg.freeze_backbone,
                image_channels=self.config.data.image_channels,
                cosmos_block_index=backbone_cfg.cosmos_block_index,
                cosmos_model_size=backbone_cfg.cosmos_model_size,
                cosmos_resolution=backbone_cfg.cosmos_resolution,
                cosmos_fps=backbone_cfg.cosmos_fps,
                cosmos_aspect_ratio=backbone_cfg.cosmos_aspect_ratio,
                cosmos_natten=backbone_cfg.cosmos_natten,
                cosmos_dit_path=backbone_cfg.cosmos_dit_path,
                cosmos_checkpoints_root=backbone_cfg.cosmos_checkpoints_root,
                cosmos_auto_fetch_checkpoints=backbone_cfg.cosmos_auto_fetch_checkpoints,
                cosmos_default_prompt=backbone_cfg.cosmos_default_prompt,
                cosmos_num_conditional_frames=backbone_cfg.cosmos_num_conditional_frames,
                cosmos_intermediate_pool=backbone_cfg.cosmos_intermediate_pool,
            )
        )

    def _build_model(self) -> ActionDiT:
        model_cfg = self.config.model
        return ActionDiT(
            ActionDiTConfig(
                action_dim=model_cfg.action_dim,
                action_chunk=model_cfg.action_chunk,
                proprio_dim=model_cfg.proprio_dim,
                vis_feature_dim=model_cfg.vis_feature_dim,
                model_dim=model_cfg.model_dim,
                depth=model_cfg.depth,
                num_heads=model_cfg.num_heads,
                mlp_ratio=model_cfg.mlp_ratio,
                dropout=model_cfg.dropout,
                text_vocab_size=model_cfg.text_vocab_size,
                max_text_tokens=model_cfg.max_text_tokens,
            )
        )

    def _build_future_source(self):
        source_cfg = self.config.future_source
        current_only = CurrentOnlyFutureSource()
        oracle_clean = OracleCleanFutureSource()
        oracle_noised = OracleNoisedFutureSource(
            OracleNoisedConfig(
                noise_std=source_cfg.noise_std,
                token_dropout_prob=source_cfg.token_dropout_prob,
            )
        )

        if source_cfg.name == "current_only":
            return current_only
        if source_cfg.name == "oracle_clean":
            return oracle_clean
        if source_cfg.name == "oracle_noised":
            return oracle_noised
        if source_cfg.name == "generated":
            return GeneratedFutureSource()
        if source_cfg.name == "mixed":
            return MixedFutureSource(
                sources={
                    "current_only": current_only,
                    "oracle_clean": oracle_clean,
                    "oracle_noised": oracle_noised,
                },
                probabilities=source_cfg.mix_probabilities,
            )
        raise ValueError(f"Unknown future source: {source_cfg.name}")

    def iter_batches(self) -> Iterator[dict[str, Any]]:
        while True:
            for batch in self.dataloader:
                yield batch

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def build_conditioning(self, batch: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        curr_vis = self.backbone.encode_current_frames(batch["current_frames"])
        future_output = self.future_source.get_future_condition(
            batch=batch,
            backbone=self.backbone,
            mode="train",
            step=self.step,
            rng=self.rng,
        )

        conditioning = {
            "curr_vis": curr_vis,
            "future_vis": future_output.future_vis,
            "future_mask": future_output.future_mask,
            "proprio": batch["proprio"],
            "text": batch["task_text"],
            "future_source_id": future_output.future_source_id,
        }
        return conditioning, future_output.metadata

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        self.model.train()
        self.backbone.train(not self.config.backbone.freeze_backbone)

        batch = self._move_batch_to_device(batch)
        conditioning, source_metadata = self.build_conditioning(batch)
        output = compute_action_flow_matching_loss(
            flow_matcher=self.flow_matcher,
            model=self.model,
            actions=batch["actions"],
            conditioning=conditioning,
            rng=self.rng,
        )

        self.optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        if self.config.train.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(
                list(self.backbone.parameters()) + list(self.model.parameters()),
                max_norm=self.config.train.grad_clip_norm,
            )
        self.optimizer.step()

        metrics = {
            "loss": float(output.loss.item()),
            "t_mean": float(output.timesteps.mean().item()),
            "future_mask_frac": float(conditioning["future_mask"].float().mean().item()),
        }
        usage = source_metadata.get("source_usage")
        if usage:
            for key, value in usage.items():
                metrics[f"source_{key}"] = float(value)
        return metrics

    @torch.no_grad()
    def sample_debug_actions(self, batch: dict[str, Any]) -> torch.Tensor:
        self.model.eval()
        batch = self._move_batch_to_device(batch)
        conditioning, _ = self.build_conditioning(batch)
        return sample_action_chunk(
            flow_matcher=self.flow_matcher,
            model=self.model,
            conditioning=conditioning,
            action_chunk=self.config.model.action_chunk,
            action_dim=self.config.model.action_dim,
            num_steps=self.config.train.flow_sample_steps,
        )

    def extract_debug_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = self._move_batch_to_device(batch)
        conditioning, metadata = self.build_conditioning(batch)
        sampled_actions = self.sample_debug_actions(batch)
        return {
            "batch_actions_shape": tuple(batch["actions"].shape),
            "curr_vis_shape": tuple(conditioning["curr_vis"].shape),
            "future_vis_shape": tuple(conditioning["future_vis"].shape),
            "future_mask_sum": int(conditioning["future_mask"].sum().item()),
            "future_source_ids": conditioning["future_source_id"].tolist(),
            "sampled_actions_shape": tuple(sampled_actions.shape),
            "source_metadata": metadata,
            "config_name": self.config.experiment.name,
            "data_config": asdict(self.config.data),
        }

    def train(self) -> None:
        batch_iter = self.iter_batches()
        for step in range(self.config.train.max_steps):
            self.step = step
            batch = next(batch_iter)
            metrics = self.train_step(batch)

            if step % self.config.train.log_every == 0 or step == self.config.train.max_steps - 1:
                self.logger.log(step=step, metrics=metrics)

            if step > 0 and step % self.config.train.sample_every == 0:
                sample_batch = next(batch_iter)
                sampled_actions = self.sample_debug_actions(sample_batch)
                self.logger.log(
                    step=step,
                    metrics={"sample_abs_mean": float(sampled_actions.abs().mean().item())},
                )
