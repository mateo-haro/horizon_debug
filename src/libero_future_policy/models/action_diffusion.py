from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class FlowMatchingOutput:
    timesteps: torch.Tensor
    noisy_actions: torch.Tensor
    target_velocity: torch.Tensor
    predicted_velocity: torch.Tensor
    loss: torch.Tensor


class ActionFlowMatcher:
    """Rectified-flow style training and sampling on action chunks."""

    def build_training_pair(
        self,
        actions: torch.Tensor,
        rng: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = actions.shape[0]
        timesteps = torch.rand(batch_size, device=actions.device, generator=rng)
        noise = torch.randn(
            actions.shape,
            device=actions.device,
            dtype=actions.dtype,
            generator=rng,
        )
        t = timesteps.view(batch_size, 1, 1)
        noisy_actions = (1.0 - t) * actions + t * noise
        target_velocity = noise - actions
        return timesteps, noisy_actions, target_velocity

    def training_loss(
        self,
        model: Any,
        actions: torch.Tensor,
        conditioning: dict[str, Any],
        rng: torch.Generator | None = None,
    ) -> FlowMatchingOutput:
        timesteps, noisy_actions, target_velocity = self.build_training_pair(actions, rng=rng)
        predicted_velocity = model(noisy_actions, timesteps, conditioning)
        loss = F.mse_loss(predicted_velocity, target_velocity)
        return FlowMatchingOutput(
            timesteps=timesteps,
            noisy_actions=noisy_actions,
            target_velocity=target_velocity,
            predicted_velocity=predicted_velocity,
            loss=loss,
        )

    @torch.no_grad()
    def sample(
        self,
        model: Any,
        conditioning: dict[str, Any],
        batch_size: int,
        action_chunk: int,
        action_dim: int,
        device: torch.device,
        num_steps: int = 16,
    ) -> torch.Tensor:
        actions = torch.randn(batch_size, action_chunk, action_dim, device=device)
        times = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        for step in range(num_steps):
            t_curr = torch.full((batch_size,), times[step].item(), device=device)
            velocity = model(actions, t_curr, conditioning)
            dt = times[step + 1] - times[step]
            actions = actions + dt * velocity
        return actions
