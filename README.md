# horizon

Research-grade LIBERO training scaffold for future-conditioned action policies.

This repo owns:

- LIBERO-style dataset window sampling
- future-conditioning abstractions
- a robotics-specific ActionDiT for action chunks
- a flow-matching training path
- training and evaluation loops

This repo does not vendor Cosmos Predict2 or the official `facebookresearch/DiT` implementation. Cosmos is treated as an external dependency behind a wrapper in `src/libero_future_policy/backbones/cosmos_adapter.py`.

## Architecture

The first working path trains an action flow-matching transformer on action chunks conditioned on:

- current visual features
- future visual features
- optional proprio
- optional task text
- future source identity

The conditioning contract is source-agnostic:

```python
{
    "curr_vis": Tensor[B, T_curr, D_vis],
    "future_vis": Tensor[B, T_future, D_vis],
    "future_mask": BoolTensor[B, T_future],
    "proprio": Optional[Tensor[B, T_curr, D_prop]],
    "text": list[str],
    "future_source_id": LongTensor[B],
}
```

Supported now:

- `current_only`
- `oracle_clean`
- `oracle_noised`
- `mixed` over the above

Stubbed for later:

- `generated`
- intermediate Cosmos denoiser-state features

## Repo Tree

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs
│   ├── data
│   ├── experiment
│   ├── model
│   └── train
├── scripts
│   ├── extract_debug_batch.py
│   ├── train_mixed_future.py
│   └── train_oracle_future.py
├── src
│   └── libero_future_policy
│       ├── backbones
│       ├── conditioning
│       ├── data
│       ├── future_sources
│       ├── models
│       ├── train
│       └── utils
└── tests
```

## Quickstart

```bash
python -m pip install -e .
python scripts/train_oracle_future.py --max-steps 10
python scripts/train_mixed_future.py --max-steps 10
python scripts/extract_debug_batch.py
pytest
```

By default the scripts use a synthetic LIBERO-like dataset and a lightweight visual feature extractor so the full path is runnable without Cosmos.

## Real Cosmos Integration

`CosmosAdapter` already defines the boundary that the rest of the repo uses:

- `encode_current_frames(...)`
- `encode_future_frames(...)`
- `maybe_generate_future_features(...)`
- `maybe_extract_intermediate_features(...)`

To attach real Cosmos Predict2 later, implement those methods with the external runtime and keep the output shape contract unchanged.

## Training Stages

1. Stage 1: train with `oracle_clean` and `oracle_noised`.
2. Stage 2: mix `current_only`, `oracle_clean`, and `oracle_noised`.
3. Stage 3: connect frozen Cosmos generation in `generated_future.py`.
4. Stage 4: compare clean, noised, generated, and intermediate denoiser-state features.

## Notes

- The current ActionDiT is compact and robotics-specific. It is DiT-inspired but not a copy of the original image-latent training code.
- Flow matching is used instead of DDPM training.
- Current and future frames share the same visual encoder path.

## Next Engineering Steps

1. Replace the mock visual extractor inside `CosmosAdapter` with real Cosmos feature extraction while preserving `Tensor[B, T, D_vis]` outputs.
2. Freeze the external Cosmos backbone and validate feature parity between `encode_current_frames` and `encode_future_frames`.
3. Implement `maybe_generate_future_features(...)` so it returns stochastic generated future features online, not a single cached bank.
4. Add `GeneratedFutureSource` to `mixed_future.py` with configurable per-sample mixing probabilities.
5. Add optional multi-sample generated futures per training sample and either pool or randomly choose one conditioning instance.
6. Add intermediate denoiser-state extraction in `maybe_extract_intermediate_features(...)` and expose it as another future-source mode for ablations.
