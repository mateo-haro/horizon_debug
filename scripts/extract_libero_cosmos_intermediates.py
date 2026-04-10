"""Extract Cosmos DiT intermediate features over sliding windows on LIBERO Cosmos MP4 data.

Expects the VideoDataset layout from ``cosmos-predict2/scripts/prepare_libero_cosmos_dataset.py``::

    <dataset-root>/
        videos/<episode>.mp4
        metas/<episode>.txt   # task caption (optional; fallback prompt if missing)

Defaults match ``cosmos-predict2/scripts/eval_libero_cosmos.py`` (480p, 10 fps, 1:1, 5 conditional
frames, 2B). Temporal window defaults to **93** frames (Libero Cosmos / Video2World training length).

**Frame indices** are **0-based**. For an episode with ``T`` frames and window length ``W``, windows
are ``[start, start+W)`` for ``start = 0 … T - W`` (inclusive start, exclusive end). If ``T < W``,
the episode is skipped with a warning.

**Noise** (``--noise-mode``):

- ``fixed``: pass ``--denoise-level`` in ``[0, 1]`` to the adapter every window.
- ``scheduler``: random ``sigma`` from the training scheduler each window (``denoise_level=None``).
- ``uniform``: sample ``denoise_level`` uniformly in ``[--denoise-min, --denoise-max]`` per window.

Each window is saved under
``--out-dir/<episode_stem>/window_<start>_<end>_<noise_tag>.pt`` (end exclusive). ``noise_tag`` is
``dl`` + denoise level (e.g. ``dl0p500000``) when a level was passed, or ``sg`` + effective ``sigma``
when the scheduler sampled noise (``denoise_level`` is ``None``). Payload includes the same fields as before.

Run from repo root (requires CUDA, cosmos-predict2, checkpoints)::

    python scripts/extract_libero_cosmos_intermediates.py \\
        --dataset-root datasets/libero_cosmos_mp4/val \\
        --out-dir outputs/cosmos_intermediates \\
        --scope all

Single episode (``--dataset-root`` is the split directory containing ``videos/`` and ``metas/``;
``--episode-stem`` is the MP4 filename **without** ``.mp4``)::

    # Episode file:
    #   /data/cosmos-predict2/datasets/libero_cosmos_mp4/train/videos/episode_data--suite=libero_spatial--2025_08_03-18_55_42--task=9--ep=500--success=True--regen_demo.mp4
    python scripts/extract_libero_cosmos_intermediates.py \\
        --dataset-root /data/cosmos-predict2/datasets/libero_cosmos_mp4/train \\
        --out-dir outputs/cosmos_intermediates \\
        --scope episode \\
        --episode-stem 'episode_data--suite=libero_spatial--2025_08_03-18_55_42--task=9--ep=500--success=True--regen_demo'

Environment: ``TOKENIZERS_PARALLELISM=false`` is set to avoid Hugging Face tokenizer warnings.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import random

import numpy as np
import torch
from decord import VideoReader, cpu
from tqdm import tqdm

from libero_future_policy.backbones.cosmos_adapter import CosmosAdapter, CosmosAdapterConfig

_DEFAULT_PROMPT = "robot manipulation task"


def _check_dataset_root(root: Path) -> Path:
    root = root.resolve()
    vdir = root / "videos"
    if not vdir.is_dir():
        raise SystemExit(f"Missing videos/ under dataset root: {root}")
    return root


def _read_prompt(metas_dir: Path, stem: str) -> str:
    p = metas_dir / f"{stem}.txt"
    if p.is_file():
        return p.read_text().strip() or _DEFAULT_PROMPT
    return _DEFAULT_PROMPT


def _filter_episodes(
    dataset_root: Path,
    scope: str,
    *,
    task_contains: str | None,
    task_regex: str | None,
    episode_stem: str | None,
    episode_glob: str | None,
) -> list[Path]:
    videos = sorted((dataset_root / "videos").glob("*.mp4"))
    metas_dir = dataset_root / "metas"
    out: list[Path] = []
    compiled: re.Pattern[str] | None = re.compile(task_regex) if task_regex else None

    for vp in videos:
        stem = vp.stem
        if scope == "episode":
            if episode_stem is not None and stem != episode_stem:
                continue
            if episode_glob is not None and not fnmatch.fnmatch(stem, episode_glob):
                continue
        if scope == "task":
            cap = _read_prompt(metas_dir, stem)
            if task_contains is not None and task_contains not in cap:
                continue
            if compiled is not None and compiled.search(cap) is None:
                continue
        out.append(vp)
    return out


def _decord_batch_to_tensor(batch: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    """decord ``get_batch`` output -> ``[1, C, T, H, W]`` float in ``[0, 1]`` on ``device``."""
    arr = np.asarray(batch.asnumpy() if hasattr(batch, "asnumpy") else batch)
    t = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous().unsqueeze(0)
    t = t.to(device=device, dtype=torch.float32)
    if t.max() > 1.0:
        t = t / 255.0
    return t.clamp(0.0, 1.0)


def _noise_tag_denoise_level(denoise_level: float) -> str:
    """Filesystem-safe tag for a fixed ``denoise_level`` in ``[0, 1]`` (must match adapter metadata)."""
    s = f"{float(denoise_level):.6f}".replace(".", "p").replace("-", "m")
    return f"dl{s}"


def _noise_tag_from_meta(meta: dict) -> str:
    """Tag from saved metadata: ``dl…`` when ``denoise_level`` is set, else ``sg…`` for ``sigma_b[0]``."""
    dl = meta["denoise_level"]
    if dl is not None:
        return _noise_tag_denoise_level(float(dl))
    sig = float(meta["sigma_b"][0])
    s = f"{sig:.6f}".replace(".", "p").replace("-", "m")
    return f"sg{s}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sliding-window Cosmos intermediate features for LIBERO MP4 dataset.")
    p.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to train/ or val/ (must contain videos/ and optionally metas/).",
    )
    p.add_argument("--out-dir", type=Path, required=True, help="Output root; one subfolder per episode.")
    p.add_argument("--scope", choices=("all", "task", "episode"), default="all")
    p.add_argument("--task-contains", type=str, default=None, help="With --scope task: caption substring.")
    p.add_argument("--task-regex", type=str, default=None, help="With --scope task: regex on caption.")
    p.add_argument("--episode-stem", type=str, default=None, help="With --scope episode: exact episode name (no .mp4).")
    p.add_argument("--episode-glob", type=str, default=None, help="With --scope episode: fnmatch on stem.")
    p.add_argument("--window-frames", type=int, default=93, help="Temporal length W (default 93).")
    p.add_argument("--noise-mode", choices=("fixed", "scheduler", "uniform"), default="fixed")
    p.add_argument("--denoise-level", type=float, default=0.5, help="For fixed mode: [0, 1].")
    p.add_argument("--denoise-min", type=float, default=0.0, help="For uniform mode.")
    p.add_argument("--denoise-max", type=float, default=1.0, help="For uniform mode.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (uniform noise + torch).")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a window if the expected output file already exists (exact path; fixed/uniform only before GPU).",
    )
    p.add_argument("--feature-dim", type=int, default=128)
    p.add_argument("--block", type=int, default=0, help="DiT block index k.")
    p.add_argument("--pool", type=str, choices=("mean", "none"), default="mean")
    p.add_argument("--dit-path", type=str, default=None)
    p.add_argument("--checkpoints-root", type=str, default=None)
    p.add_argument("--model-size", type=str, default="2B")
    p.add_argument("--resolution", type=str, default="480")
    p.add_argument("--fps", type=int, default=10, choices=(10, 16))
    p.add_argument("--aspect-ratio", type=str, default="1:1")
    p.add_argument("--natten", action="store_true")
    p.add_argument("--num-conditional-frames", type=int, default=5)
    p.add_argument("--conditioning-strategy", type=str, default="frame_replace")
    p.add_argument("--gpu-text-encoder", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.scope == "task" and args.task_contains is None and args.task_regex is None:
        raise SystemExit("--scope task requires --task-contains and/or --task-regex")
    if args.scope == "episode" and args.episode_stem is None and args.episode_glob is None:
        raise SystemExit("--scope episode requires --episode-stem and/or --episode-glob")
    if args.noise_mode == "uniform" and args.denoise_min > args.denoise_max:
        raise SystemExit("--denoise-min must be <= --denoise-max")

    dataset_root = _check_dataset_root(args.dataset_root)
    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    episodes = _filter_episodes(
        dataset_root,
        args.scope,
        task_contains=args.task_contains,
        task_regex=args.task_regex,
        episode_stem=args.episode_stem,
        episode_glob=args.episode_glob,
    )
    if not episodes:
        raise SystemExit("No episodes matched filters.")

    adapter = CosmosAdapter(
        CosmosAdapterConfig(
            feature_dim=args.feature_dim,
            use_external_cosmos=True,
            external_cosmos_module=None,
            freeze_backbone=True,
            image_channels=3,
            cosmos_block_index=args.block,
            cosmos_model_size=args.model_size,
            cosmos_resolution=args.resolution,
            cosmos_fps=args.fps,
            cosmos_aspect_ratio=args.aspect_ratio,
            cosmos_natten=args.natten,
            cosmos_dit_path=args.dit_path,
            cosmos_checkpoints_root=args.checkpoints_root,
            cosmos_default_prompt="",
            cosmos_num_conditional_frames=args.num_conditional_frames,
            cosmos_conditioning_strategy=args.conditioning_strategy,
            cosmos_intermediate_pool=args.pool,
            cosmos_offload_text_encoder=not args.gpu_text_encoder,
            cosmos_verbose_load=args.verbose,
        )
    )
    if getattr(adapter, "_cosmos_pipe", None) is None:
        reason = getattr(adapter, "cosmos_pipeline_load_error", None)
        raise SystemExit(f"Cosmos pipeline did not load. {reason or ''}")

    device = torch.device("cuda")
    W = args.window_frames
    rng = random.Random(args.seed)
    metas_dir = dataset_root / "metas"

    for video_path in tqdm(episodes, desc="episodes"):
        stem = video_path.stem
        prompt = _read_prompt(metas_dir, stem)
        caption_path = metas_dir / f"{stem}.txt"

        vr = VideoReader(str(video_path), ctx=cpu(0))
        n_frames = len(vr)
        if n_frames < W:
            print(f"skip {stem}: only {n_frames} frames (< window {W})", file=sys.stderr)
            continue

        ep_dir = out_root / stem
        ep_dir.mkdir(parents=True, exist_ok=True)

        for start in tqdm(
            range(0, n_frames - W + 1),
            desc=f"windows {stem}",
            leave=False,
        ):
            end_excl = start + W

            if args.noise_mode == "fixed":
                denoise_level: float | None = float(args.denoise_level)
            elif args.noise_mode == "scheduler":
                denoise_level = None
            else:
                denoise_level = rng.uniform(args.denoise_min, args.denoise_max)

            if args.skip_existing and args.noise_mode != "scheduler" and denoise_level is not None:
                tag_pre = _noise_tag_denoise_level(float(denoise_level))
                cand = ep_dir / f"window_{start:06d}_{end_excl:06d}_{tag_pre}.pt"
                if cand.is_file():
                    continue

            batch = vr.get_batch(list(range(start, start + W)))
            frames = _decord_batch_to_tensor(batch, device)
            result = adapter.maybe_extract_intermediate_features(
                frames,
                denoise_level=denoise_level,
                prompt=prompt,
                return_metadata=True,
            )
            if result is None:
                print(f"warning: None features for {stem} window {start}-{end_excl}", file=sys.stderr)
                continue
            feats, meta = result

            noise_tag = _noise_tag_from_meta(meta)
            out_path = ep_dir / f"window_{start:06d}_{end_excl:06d}_{noise_tag}.pt"
            if args.skip_existing and args.noise_mode == "scheduler" and out_path.is_file():
                continue

            payload = {
                "features": feats.detach().cpu(),
                "prompt": meta["prompt"],
                "denoise_level": meta["denoise_level"],
                "sigma_b": meta["sigma_b"],
                "start_frame": start,
                "end_frame": end_excl,
                "episode_stem": stem,
                "video_path": str(video_path.resolve()),
                "caption_path": str(caption_path.resolve()) if caption_path.is_file() else None,
                "noise_mode": args.noise_mode,
                "window_frames": W,
                "cosmos_intermediate_pool": args.pool,
                "cosmos_block_index": args.block,
                "noise_tag": noise_tag,
                "output_filename": out_path.name,
            }
            torch.save(payload, out_path)

    print(f"Done. Outputs under {out_root}")


if __name__ == "__main__":
    main()
