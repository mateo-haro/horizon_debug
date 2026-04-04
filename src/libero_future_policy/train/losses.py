from __future__ import annotations

from typing import Any

from libero_future_policy.models.action_diffusion import ActionFlowMatcher, FlowMatchingOutput


def compute_action_flow_matching_loss(
    flow_matcher: ActionFlowMatcher,
    model: Any,
    actions,
    conditioning,
    rng=None,
) -> FlowMatchingOutput:
    return flow_matcher.training_loss(model=model, actions=actions, conditioning=conditioning, rng=rng)
