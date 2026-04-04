from __future__ import annotations

from typing import Any

import torch

from libero_future_policy.models.action_diffusion import ActionFlowMatcher


@torch.no_grad()
def sample_action_chunk(
    flow_matcher: ActionFlowMatcher,
    model: Any,
    conditioning: dict[str, Any],
    action_chunk: int,
    action_dim: int,
    num_steps: int,
) -> torch.Tensor:
    curr_vis = conditioning["curr_vis"]
    return flow_matcher.sample(
        model=model,
        conditioning=conditioning,
        batch_size=curr_vis.shape[0],
        action_chunk=action_chunk,
        action_dim=action_dim,
        device=curr_vis.device,
        num_steps=num_steps,
    )
