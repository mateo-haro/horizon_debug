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

`CosmosAdapter` defines the boundary that the rest of the repo uses:

- `encode_current_frames(...)`
- `encode_future_frames(...)`
- `maybe_generate_future_features(...)`
- `maybe_extract_intermediate_features(...)`

To run the **Hugging Face Diffusers** snippet (`Cosmos2VideoToWorldPipeline`, see `src/libero_future_policy/backbones/cosmos_test.py`), install **diffusers >= 0.34** (Horizon does not pin it by default): `pip install -U "diffusers>=0.34"` or `pip install -e ".[diffusers_cosmos]"`.

Install Predict2 into the same environment (Python 3.10, CUDA) using NVIDIA’s index, for example:

```bash
uv pip install -U "cosmos-predict2[cu126]" --extra-index-url https://nvidia-cosmos.github.io/cosmos-dependencies/cu126_torch260/simple
```

If imports fail with Transformer Engine / ``ldconfig`` / ``libnvrtc``, `CosmosAdapter` tries to set ``CUDA_PATH`` to pip’s ``nvidia-cuda-nvrtc`` layout before loading Cosmos. Disable that with ``HORIZON_SKIP_NVRTC_BOOTSTRAP=1`` if it conflicts with your CUDA layout.

Default Predict2 Video2World weights and the T5-11B text encoder are resolved under a checkpoints **root** (parent of ``nvidia/`` and ``google-t5/``): set ``COSMOS_CHECKPOINTS_DIR``, YAML ``cosmos_checkpoints_root``, or use the default ``~/.cache/horizon/cosmos_checkpoints``. With ``cosmos_auto_fetch_checkpoints: true`` (default), missing files are downloaded from Hugging Face (``nvidia/Cosmos-Predict2-{size}-Video2World`` and ``google-t5/t5-11b``) the first time a CUDA pipeline is built.

Enable it in YAML under `backbone` with `use_external_cosmos: true` and set checkpoint-related options (`cosmos_dit_path`, `cosmos_model_size`, `cosmos_resolution`, `cosmos_fps`, `cosmos_aspect_ratio`, `cosmos_natten`) as needed. Weights are loaded from the Cosmos checkpoint at startup and are not stored in Horizon checkpoints.

`maybe_extract_intermediate_features` runs a single Video2World denoise step, captures the DiT activation **after** block `cosmos_block_index`, and returns either a mean-pooled `[B, feature_dim]` vector (with an optional linear projection) or the full patch grid `[B, T', H', W', D]` when `cosmos_intermediate_pool: none`. It requires **CUDA** and input `frames` on **cuda**; otherwise it returns `None` (CPU training stays valid).

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
