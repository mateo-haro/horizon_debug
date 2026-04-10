# Cosmos backbone (Horizon)

This folder wraps **Cosmos Predict2 Video2World** behind [`cosmos_adapter.py`](cosmos_adapter.py) so the rest of the repo does not import `cosmos_predict2` directly. Two scripts exercise the same adapter:

| Script | Purpose |
|--------|---------|
| [`scripts/cosmos_intermediate_demo.py`](../../../scripts/cosmos_intermediate_demo.py) | Quick smoke test: **random RGB video** on GPU → one forward → intermediate features |
| [`scripts/extract_libero_cosmos_intermediates.py`](../../../scripts/extract_libero_cosmos_intermediates.py) | **LIBERO MP4 dataset**: sliding windows → one forward per window → save `.pt` files |

**Requirements:** NVIDIA GPU, CUDA PyTorch, `cosmos-predict2` + checkpoints (Video2World DiT, tokenizer, T5). First-time downloads usually go under `COSMOS_CHECKPOINTS_DIR` or `~/.cache/horizon/cosmos_checkpoints` unless you pass `--checkpoints-root` / `--dit-path`.

---

## `CosmosAdapter` and `CosmosAdapterConfig`

`CosmosAdapter` loads `Video2WorldPipeline` when `use_external_cosmos=True`, freezes the DiT and (by default) T5, and exposes:

- **`maybe_extract_intermediate_features(frames, denoise_level=..., prompt=..., return_metadata=False)`** — Runs **one** `pipe.denoise(...)` forward (not full multi-step sampling). Registers a forward hook on DiT block `cosmos_block_index`, corrupts latents with `x_t = x_0 + σ·ε`, and returns either mean-pooled features (optional linear projection to `feature_dim`) or the full patch grid when `cosmos_intermediate_pool="none"`. Inputs must be **`frames` on CUDA** with shape **`[B, C, T, H, W]`** (uint8 or float in `[0,1]` / `[-1,1]`).

### Config fields (`CosmosAdapterConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `feature_dim` | *(required)* | Target dimension when pooling is `"mean"` and a projection is created from DiT width → `feature_dim`. |
| `use_external_cosmos` | `False` | Must be **`True`** to load the Video2World pipeline. |
| `external_cosmos_module` | `None` | Optional import hook; usually `None`. |
| `freeze_backbone` | `False` | If `True`, no gradients on encoder, projection, text encoder, DiT. |
| `image_channels` | `3` | RGB. |
| `cosmos_block_index` | `0` | Which DiT block output to capture (clamped to valid range). |
| `cosmos_model_size` | `"2B"` | Checkpoint family (e.g. `2B`). |
| `cosmos_resolution` | `"480"` | Must match weights (e.g. `480`, `720`). |
| `cosmos_fps` | `16` | Must match checkpoint (`10` or `16`). |
| `cosmos_aspect_ratio` | `"1:1"` | Must exist in Cosmos `VIDEO_RES_SIZE_INFO` for your resolution. |
| `cosmos_natten` | `False` | NATTEN variant; aspect ratio restrictions apply in upstream. |
| `cosmos_dit_path` | `None` | Override path to DiT `.pt`; if `None`, resolved via `imaginaire.constants`. |
| `cosmos_checkpoints_root` | `None` | Root containing `nvidia/...` and `google-t5/...` layouts. |
| `cosmos_auto_fetch_checkpoints` | `True` | Hugging Face download when weights missing (if implemented). |
| `cosmos_default_prompt` | `""` | Used when `prompt=None` in feature extraction. |
| `cosmos_num_conditional_frames` | `5` | First *N* time steps treated as conditioning (frame-replace strategy). |
| `cosmos_conditioning_strategy` | `"frame_replace"` | `"frame_replace"` keeps first N frames clean in `denoise`; `"channel_concat"` is the alternative upstream mode. |
| `cosmos_intermediate_pool` | `"none"` | `"mean"` → global average over space-time then optional projection; `"none"` → raw block output grid. |
| `cosmos_offload_text_encoder` | `True` | Keep T5 on CPU until encode (lower peak VRAM at init). |
| `cosmos_downcast_text_encoder` | `True` | Dtype for T5. |
| `cosmos_verbose_load` | `False` | stderr logs during load; or set `HORIZON_COSMOS_DEBUG=1`. |

With `return_metadata=True`, `maybe_extract_intermediate_features` also returns a dict with `prompt`, `denoise_level` (argument, or `None` if the scheduler sampled σ), and `sigma_b` (per-batch effective σ after pipeline scaling).

---

## Script: `cosmos_intermediate_demo.py`

**What it does:** Builds a **dummy** tensor `[1,3,T,H,W]` on CUDA, constructs `CosmosAdapter` with CLI flags, calls `maybe_extract_intermediate_features` once, prints feature shape.

Run from **repository root**:

```bash
python scripts/cosmos_intermediate_demo.py
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--block` | `0` | DiT block index (same as `cosmos_block_index`). |
| `--feature-dim` | `128` | `CosmosAdapterConfig.feature_dim`. |
| `--denoise-level` | `0.5` | Interpolant in `[0,1]` on scheduler σ range. **Ignored** if `--random-sigma`. |
| `--random-sigma` | off | Sample σ from the training scheduler instead of fixed level. |
| `--prompt` | *(long default string)* | Text prompt for T5. |
| `--dit-path` | `None` | Local DiT checkpoint path. |
| `--checkpoints-root` | `None` | Checkpoints root directory. |
| `--pool` | `mean` | `mean` or `none` → `cosmos_intermediate_pool`. |
| `--model-size` | `2B` | Model size. |
| `--resolution` | `720` | e.g. `480`, `720`. |
| `--fps` | `16` | `10` or `16`. |
| `--aspect-ratio` | `16:9` | Must be valid for resolution. |
| `--natten` | off | NATTEN pipeline. |
| `--time-frames` | `8` | Dummy sequence length *before* adapter resize. |
| `--height`, `--width` | `64` | Dummy spatial size *before* adapter resize. |
| `--require-success` | off | Exit `1` if CUDA/pipeline/extraction fails (default exits `0` on skip). |
| `--verbose` | off | Load stages + CUDA memory on stderr. |
| `--gpu-text-encoder` | off | Keep T5 on GPU during init (more VRAM). |

### Examples

```bash
# Default quick run (may download checkpoints first time)
python scripts/cosmos_intermediate_demo.py

# Libero-eval-like model settings + fixed denoise level
python scripts/cosmos_intermediate_demo.py \
  --resolution 480 --fps 10 --aspect-ratio 1:1 \
  --denoise-level 0.3 --pool mean --verbose

# Random σ each run (matches training-time sampling distribution)
python scripts/cosmos_intermediate_demo.py --random-sigma --require-success

# Custom DiT weights
python scripts/cosmos_intermediate_demo.py --dit-path /path/to/model-480p-10fps.pt
```

---

## Script: `extract_libero_cosmos_intermediates.py`

**What it does:** Reads episodes from a **VideoDataset** layout (same as `cosmos-predict2/scripts/prepare_libero_cosmos_dataset.py`): `videos/*.mp4` and optional `metas/*.txt` captions. For each episode, slides a window of length `W` over **every** start index `0 … T−W` (one **batch size = 1** forward per window). Saves `torch.load`-able dicts under `--out-dir/<episode_stem>/`.

**Inference:** Not batched across windows; each window is a separate `maybe_extract_intermediate_features` call.

### Data layout

```
<dataset-root>/          # e.g. .../libero_cosmos_mp4/train
  videos/<episode>.mp4
  metas/<episode>.txt    # optional; one-line task caption
```

`--dataset-root` must be the directory that **contains** `videos/` (not the parent of `train` unless that parent also has `videos/`).

### Output files

Pattern:

`window_<start>_<end>_<noise_tag>.pt`

- `<start>` / `<end>`: **0-based** frame indices; `<end>` is **exclusive** (covers `[start, end)`).
- **noise_tag**
  - **`dl` + value**: normalized `denoise_level` in `[0,1]`. The filename uses `p` instead of `.` (e.g. `dl0p500000` → **0.5**).
  - **`sg` + value**: effective σ when the scheduler sampled noise (`denoise_level=None`); same `p`/`m` encoding.

Each payload includes `features`, `prompt`, `denoise_level`, `sigma_b`, frame indices, paths, `noise_tag`, `output_filename`, etc.

### Arguments

**Dataset & scope**

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset-root` | *(required)* | Path to split dir with `videos/`. |
| `--out-dir` | *(required)* | Output root. |
| `--scope` | `all` | `all` \| `task` \| `episode`. |
| `--task-contains` | `None` | With `task`: substring on caption text. |
| `--task-regex` | `None` | With `task`: regex on caption. |
| `--episode-stem` | `None` | With `episode`: exact filename stem (no `.mp4`). |
| `--episode-glob` | `None` | With `episode`: `fnmatch` on stem. |

**Window & noise**

| Argument | Default | Description |
|----------|---------|-------------|
| `--window-frames` | `93` | Window length `W` (Libero Cosmos default). Episodes with `T < W` are skipped. |
| `--noise-mode` | `fixed` | `fixed` \| `scheduler` \| `uniform`. |
| `--denoise-level` | `0.5` | Used when `noise-mode=fixed`. |
| `--denoise-min`, `--denoise-max` | `0`, `1` | Used when `noise-mode=uniform` (per-window sample). |
| `--seed` | `42` | Python + PyTorch RNG. |
| `--skip-existing` | off | Skip if output file exists (see script help for scheduler vs fixed/uniform). |

**Cosmos (aligned with `eval_libero_cosmos.py` defaults)**

| Argument | Default | Description |
|----------|---------|-------------|
| `--feature-dim` | `128` | |
| `--block` | `0` | DiT block index. |
| `--pool` | `mean` | `mean` or `none`. |
| `--dit-path` | `None` | |
| `--checkpoints-root` | `None` | |
| `--model-size` | `2B` | |
| `--resolution` | `480` | |
| `--fps` | `10` | `10` or `16`. |
| `--aspect-ratio` | `1:1` | |
| `--natten` | off | |
| `--num-conditional-frames` | `5` | |
| `--conditioning-strategy` | `frame_replace` | |
| `--gpu-text-encoder` | off | |
| `--verbose` | off | |

### Examples

```bash
# Whole val split
python scripts/extract_libero_cosmos_intermediates.py \
  --dataset-root datasets/libero_cosmos_mp4/val \
  --out-dir outputs/cosmos_intermediates \
  --scope all

# Single episode (stem = filename without .mp4)
python scripts/extract_libero_cosmos_intermediates.py \
  --dataset-root /data/cosmos-predict2/datasets/libero_cosmos_mp4/train \
  --out-dir outputs/cosmos_intermediates \
  --scope episode \
  --episode-stem 'episode_data--suite=libero_spatial--2025_08_03-18_55_42--task=9--ep=500--success=True--regen_demo'

# Tasks whose caption contains a phrase
python scripts/extract_libero_cosmos_intermediates.py \
  --dataset-root datasets/libero_cosmos_mp4/train \
  --out-dir outputs/cosmos_intermediates \
  --scope task \
  --task-contains "drawer"

# Scheduler-random σ per window
python scripts/extract_libero_cosmos_intermediates.py \
  --dataset-root datasets/libero_cosmos_mp4/val \
  --out-dir outputs/cosmos_rand \
  --scope all \
  --noise-mode scheduler
```

---

## Environment

- `TOKENIZERS_PARALLELISM=false` — set by the extract script; avoids tokenizer warnings in multi-process contexts.
- `HORIZON_COSMOS_DEBUG=1` — verbose Cosmos load logs from the adapter.
- `COSMOS_CHECKPOINTS_DIR` / `--checkpoints-root` — where Video2World + T5 weights are expected.

---

## Troubleshooting

- **Exit code 137** — often Linux OOM; try `--resolution 480`, keep text encoder offloaded (default), avoid `--gpu-text-encoder`.
- **Pipeline `None`** — check stderr / `cosmos_pipeline_load_error`; verify checkpoints and CUDA.
- **`cosmos_readme.md` in this folder** — upstream Cosmos marketing copy, not maintained for Horizon; use **this README** for adapter and scripts.
