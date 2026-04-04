from __future__ import annotations

import torch

from libero_future_policy.models.action_diffusion import ActionFlowMatcher
from libero_future_policy.models.action_dit import ActionDiT, ActionDiTConfig


def test_action_dit_forward_shapes() -> None:
    model = ActionDiT(
        ActionDiTConfig(
            action_dim=7,
            action_chunk=8,
            proprio_dim=8,
            vis_feature_dim=64,
            model_dim=128,
            depth=3,
            num_heads=4,
            mlp_ratio=4.0,
            dropout=0.0,
            text_vocab_size=1024,
            max_text_tokens=8,
        )
    )

    conditioning = {
        "curr_vis": torch.randn(2, 2, 64),
        "future_vis": torch.randn(2, 3, 64),
        "future_mask": torch.ones(2, 3, dtype=torch.bool),
        "proprio": torch.randn(2, 2, 8),
        "text": ["pick the block", "open the drawer"],
        "future_source_id": torch.tensor([1, 2], dtype=torch.long),
    }
    actions = torch.randn(2, 8, 7)
    timesteps = torch.rand(2)
    out = model(actions, timesteps, conditioning)
    assert out.shape == (2, 8, 7)


def test_flow_matching_loss_runs() -> None:
    model = ActionDiT(
        ActionDiTConfig(
            action_dim=7,
            action_chunk=8,
            proprio_dim=8,
            vis_feature_dim=64,
            model_dim=128,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            text_vocab_size=512,
            max_text_tokens=8,
        )
    )
    flow = ActionFlowMatcher()
    conditioning = {
        "curr_vis": torch.randn(2, 2, 64),
        "future_vis": torch.randn(2, 2, 64),
        "future_mask": torch.ones(2, 2, dtype=torch.bool),
        "proprio": torch.randn(2, 2, 8),
        "text": ["task one", "task two"],
        "future_source_id": torch.tensor([1, 0], dtype=torch.long),
    }
    actions = torch.randn(2, 8, 7)
    output = flow.training_loss(model, actions, conditioning)
    assert output.loss.ndim == 0
