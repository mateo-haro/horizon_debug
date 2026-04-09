from __future__ import annotations

import torch

from lerobot_policy_horizon.backbones.cosmos_adapter import CosmosAdapter


def test_cosmos_adapter_encode_and_hidden_shapes() -> None:
    adapter = CosmosAdapter(
        feature_dim=64,
        hidden_layer_index=2,
        noise_level=0.2,
        num_hidden_layers=4,
    )
    frames = torch.randn(2, 3, 3, 64, 64)

    encoded = adapter.encode(frames)
    assert encoded.shape == (2, 3, 64)

    denoise_state = adapter.denoise_to_tau(frames, tau=0.5)
    assert len(denoise_state.hidden_states) == 4
    assert denoise_state.hidden_states[0].shape == (2, 3, 64)

    hidden = adapter.get_nth_hidden_layer(denoise_state, hidden_layer_index=1)
    assert hidden.shape == (2, 3, 64)


def test_cosmos_adapter_multiview() -> None:
    adapter = CosmosAdapter(feature_dim=32, num_hidden_layers=3)
    views = [torch.randn(2, 2, 3, 32, 32), torch.randn(2, 2, 3, 32, 32)]
    features = adapter.encode_multiview(views)
    hidden = adapter.extract_hidden_features_multiview(views, tau=0.3, hidden_layer_index=0, noise_level=0.1)
    assert features.shape == (2, 2, 32)
    assert hidden.shape == (2, 2, 32)
