# Cosmos backbone (Horizon)

This folder wraps **Cosmos Predict2 Video2World** behind [`cosmos_adapter.py`](cosmos_adapter.py) so the rest of the repo does not import `cosmos_predict2` directly. Two scripts exercise the same adapter:

| Script | Purpose |
|--------|---------|
| [`scripts/cosmos_intermediate_demo.py`](../../../scripts/cosmos_intermediate_demo.py) | Quick smoke test: **random RGB video** on GPU → one forward → intermediate features |
| [`scripts/extract_libero_cosmos_intermediates.py`](../../../scripts/extract_libero_cosmos_intermediates.py) | **LIBERO MP4 dataset**: sliding windows → one forward per window → save `.pt` files |

**Requirements:** NVIDIA GPU, CUDA PyTorch, `cosmos-predict2` + checkpoints (Video2World DiT, tokenizer, T5). First-time downloads usually go under `COSMOS_CHECKPOINTS_DIR` or `~/.cache/horizon/cosmos_checkpoints` unless you pass `--checkpoints-root` / `--dit-path`.

---

## `CosmosVideoAdapter` and `CosmosAdapterConfig`

`CosmosVideoAdapter` loads `Video2WorldPipeline` when `use_external_cosmos=True`, freezes the DiT and (by default) T5, and exposes:

- **`maybe_extract_intermediate_features(frames, denoise_level=..., prompt=..., return_metadata=False)`** — Runs **one** `pipe.denoise(...)` forward (not full multi-step sampling). Registers a forward hook on DiT block `cosmos_block_index`, corrupts latents with `x_t = x_0 + σ·ε`, and returns either **mean-pooled** DiT activations (`cosmos_intermediate_pool="mean"`, last dim = DiT `model_channels`) or the **raw** block grid (`"none"`). There is **no learned `Linear` in the adapter**; map latents to `cosmos_feature_dim` in **`HorizonDiTPolicy`** via `VisualLatentProjection` (see `horizon/models/visual_latent_projection.py`). Inputs must be **`frames` on CUDA** with shape **`[B, C, T, H, W]`** (uint8 or float in `[0,1]` / `[-1,1]`). **`denoise_level`** may be **`None`** (scheduler samples one σ per batch row), a **scalar float** in `[0,1]` (same level for every row), or a **1D float tensor of shape `(B,)`** (one level per row, e.g. batched uniform noise).

### Config fields (`CosmosAdapterConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `feature_dim` | *(required)* | Used for the lightweight CNN fallback encoder and `empty_future_features`; not a learned projection target for Video2World intermediates. |
| `use_external_cosmos` | `False` | Must be **`True`** to load the Video2World pipeline. |
| `external_cosmos_module` | `None` | Optional import hook; usually `None`. |
| `freeze_backbone` | `False` | If `True`, no gradients on encoder, text encoder, DiT. |
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
| `cosmos_intermediate_pool` | `"none"` | `"mean"` → global average over space-time (DiT channel dim); `"none"` → raw block output grid. **Default in** `extract_libero_cosmos_intermediates.py` **is** `"none"` (raw). |
| `cosmos_offload_text_encoder` | `True` | Keep T5 on CPU until encode (lower peak VRAM at init). |
| `cosmos_downcast_text_encoder` | `True` | Dtype for T5. |
| `cosmos_verbose_load` | `False` | stderr logs during load; or set `HORIZON_COSMOS_DEBUG=1`. |

With `return_metadata=True`, `maybe_extract_intermediate_features` also returns a dict with `prompt`, `denoise_level` (scalar or `None` as passed in, or a **CPU float tensor `(B,)`** when a per-row tensor was passed), and `sigma_b` (per-batch effective σ after pipeline scaling, shape `(B,)` on CPU).

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

**What it does:** Reads episodes from a **VideoDataset** layout (same as `cosmos-predict2/scripts/prepare_libero_cosmos_dataset.py`): `videos/*.mp4` and optional `metas/*.txt` captions. Builds an **extended timeline** per episode: `num_conditional_frames` copies of frame **0** before the real video, then all real frames `0 … T−1`, then (if needed) copies of the **last** frame so the extended length is at least `W` when `T + num_conditional_frames < W`. It slides a window of length `W` over every virtual start `0 … L−W` (`L` = extended length). Saves one `torch.load`-able dict per window under `--out-dir/<episode_stem>/`.

**Inference:** `--batch-size` (default `1`) stacks that many windows from the **same episode** into one `maybe_extract_intermediate_features` call; the last chunk in an episode may be smaller, and the next episode starts a new chunk at the full batch size. With `noise-mode=uniform`, per-window levels are passed as a `(B,)` tensor to the adapter.

### Data layout

```
<dataset-root>/          # e.g. .../libero_cosmos_mp4/train
  videos/<episode>.mp4
  metas/<episode>.txt    # optional; one-line task caption
```

`--dataset-root` must be the directory that **contains** `videos/` (not the parent of `train` unless that parent also has `videos/`).

### Output files

Pattern:

`window_h<head>_r<real_lo>_<real_hi>_t<tail>_<noise_tag>.pt`

- **`head`**: count of window slots filled from the **leading** synthetic region (repeats of frame 0).
- **`tail`**: count of window slots filled from the **trailing** synthetic region (repeats of the last real frame).
- **`real_lo`**, **`real_hi`**: half-open range of **original video** indices covered by real frames in this window (`-1` / `-1` if the window has no real frames).
- **noise_tag**
  - **`dl` + value**: normalized `denoise_level` in `[0,1]`. The filename uses `p` instead of `.` (e.g. `dl0p500000` → **0.5**).
  - **`sg` + value**: effective σ when the scheduler sampled noise (`denoise_level=None`); same `p`/`m` encoding.

Human-readable episode hints are also stored in the payload as `notation_episode_prefix` (e.g. `-5...` for five leading synthetic frames) and `notation_episode_suffix` (e.g. `...150+40` when the last real index is 150 and 40 tail synthetic frames were appended to the episode).

Each payload includes `features`, `prompt`, `denoise_level`, `sigma_b`, `virtual_start` / `virtual_end_exclusive`, `prefix_pad_episode`, `suffix_pad_episode`, `window_head_pad`, `window_tail_pad`, `real_start`, `real_end_exclusive`, `start_frame` / `end_frame` (same as the virtual range for compatibility), `batch_size_effective`, paths, `noise_tag`, `output_filename`, etc.

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
| `--window-frames` | `93` | Window length `W` (Libero Cosmos default). |
| `--batch-size` | `1` | Windows per GPU forward (same episode only). |
| `--noise-mode` | `fixed` | `fixed` \| `scheduler` \| `uniform`. |
| `--denoise-level` | `0.5` | Used when `noise-mode=fixed`. |
| `--denoise-min`, `--denoise-max` | `0`, `1` | Used when `noise-mode=uniform` (per-window sample; batched as a tensor). |
| `--seed` | `42` | Python + PyTorch RNG. |
| `--skip-existing` | off | Skip if output exists; for `fixed`/`uniform`, skips a whole batch only when **all** candidate files in that batch already exist. |

**Cosmos (aligned with `eval_libero_cosmos.py` defaults)**

| Argument | Default | Description |
|----------|---------|-------------|
| `--feature-dim` | `128` | |
| `--block` | `0` | DiT block index. |
| `--pool` | `none` | `none` (raw grid, default) or `mean` (global mean, DiT width). |
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
