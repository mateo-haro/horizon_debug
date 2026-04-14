"""Sliding-window LIBERO Cosmos MP4 → CosmosVideoAdapter intermediate features.

Uses an extended timeline: ``num_conditional_frames`` copies of frame 0 before the episode,
optional tail copies of the last frame so ``T < W`` episodes are not skipped, then sliding
windows of length ``W``. Batched forwards per episode when ``--batch-size > 1``.

Default ``--pool`` is ``none`` (raw DiT block grid). Use ``mean`` for a single global-pooled
vector per window (DiT width, no adapter ``Linear``). Policy-side :class:`VisualLatentProjection`
maps latents to ``cosmos_feature_dim``.

Expects ``<dataset-root>/videos/*.mp4`` and optional ``metas/*.txt``. See
``src/horizon/backbones/README.md`` for layout, filename grammar, and noise tags.

Requires CUDA, cosmos-predict2, and checkpoints. Sets ``TOKENIZERS_PARALLELISM=false``.
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

from horizon.backbones.cosmos_adapter import CosmosAdapterConfig, CosmosVideoAdapter

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


def _virtual_to_decord_idx(virtual_i: int, n_prefix: int, n_video: int) -> int:
    """Map extended-timeline index to decord frame index (``n_video`` = number of real frames)."""
    if virtual_i < n_prefix:
        return 0
    if virtual_i < n_prefix + n_video:
        return virtual_i - n_prefix
    return n_video - 1


def _extended_length(n_prefix: int, n_video: int, window_frames: int) -> tuple[int, int]:
    """Return ``(L, suffix_pad)`` — extended length and count of tail synthetic frames."""
    core = n_prefix + n_video
    suffix_pad = max(0, window_frames - core)
    return core + suffix_pad, suffix_pad


def _window_decomposition(
    virtual_start: int,
    window_frames: int,
    n_prefix: int,
    n_video: int,
) -> tuple[int, int, int, int]:
    """``(head_pad, tail_pad, real_lo, real_hi_exclusive)`` for one window; ``(-1, -1)`` if no real."""
    head = tail = 0
    reals: list[int] = []
    for j in range(window_frames):
        pos = virtual_start + j
        if pos < n_prefix:
            head += 1
        elif pos < n_prefix + n_video:
            reals.append(pos - n_prefix)
        else:
            tail += 1
    if not reals:
        return head, tail, -1, -1
    return head, tail, min(reals), max(reals) + 1


def _window_basename(head: int, real_lo: int, real_hi: int, tail: int, noise_tag: str) -> str:
    return f"window_h{head:03d}_r{real_lo}_{real_hi}_t{tail:03d}_{noise_tag}.pt"


def _noise_tag_denoise_level(denoise_level: float) -> str:
    """Filesystem-safe tag for a fixed ``denoise_level`` in ``[0, 1]`` (must match adapter metadata)."""
    s = f"{float(denoise_level):.6f}".replace(".", "p").replace("-", "m")
    return f"dl{s}"


def _noise_tag_sigma(sigma: float) -> str:
    s = f"{float(sigma):.6f}".replace(".", "p").replace("-", "m")
    return f"sg{s}"


def _noise_tag_from_meta(meta: dict, batch_index: int = 0) -> str:
    """Tag from metadata after a forward (supports batched ``denoise_level`` tensor)."""
    dl = meta["denoise_level"]
    if isinstance(dl, torch.Tensor):
        flat = dl.reshape(-1)
        if flat.numel() == 1:
            return _noise_tag_denoise_level(float(flat[0].item()))
        return _noise_tag_denoise_level(float(flat[batch_index].item()))
    if dl is not None:
        return _noise_tag_denoise_level(float(dl))
    sb = meta["sigma_b"]
    sig = float(sb[batch_index].item()) if sb.numel() > 1 else float(sb[0].item())
    return _noise_tag_sigma(sig)


def _payload_denoise_level(meta: dict, batch_index: int) -> float | None:
    dl = meta["denoise_level"]
    if isinstance(dl, torch.Tensor):
        flat = dl.reshape(-1)
        if flat.numel() == 1:
            return float(flat[0].item())
        return float(flat[batch_index].item())
    return dl


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
    p.add_argument("--batch-size", type=int, default=1, help="Windows per forward (same episode only; default 1).")
    p.add_argument("--noise-mode", choices=("fixed", "scheduler", "uniform"), default="fixed")
    p.add_argument("--denoise-level", type=float, default=0.5, help="For fixed mode: [0, 1].")
    p.add_argument("--denoise-min", type=float, default=0.0, help="For uniform mode.")
    p.add_argument("--denoise-max", type=float, default=1.0, help="For uniform mode.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (uniform noise + torch).")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip windows whose output file exists; fixed/uniform skip whole batch if all exist.",
    )
    p.add_argument("--feature-dim", type=int, default=128)
    p.add_argument("--block", type=int, default=0, help="DiT block index k.")
    p.add_argument(
        "--pool",
        type=str,
        choices=("mean", "none"),
        default="none",
        help="none=raw DiT block grid (default); mean=global mean over spatiotemporal dims (DiT width, no adapter Linear).",
    )
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
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

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

    adapter = CosmosVideoAdapter(
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
    n_prefix = max(0, int(args.num_conditional_frames))
    rng = random.Random(args.seed)
    metas_dir = dataset_root / "metas"
    batch_size = int(args.batch_size)

    for video_path in tqdm(episodes, desc="episodes"):
        stem = video_path.stem
        prompt = _read_prompt(metas_dir, stem)
        caption_path = metas_dir / f"{stem}.txt"

        vr = VideoReader(str(video_path), ctx=cpu(0))
        T = len(vr)
        if T == 0:
            print(f"skip {stem}: empty video", file=sys.stderr)
            continue

        L, suffix_pad = _extended_length(n_prefix, T, W)
        virtual_starts = list(range(0, L - W + 1))

        ep_dir = out_root / stem
        ep_dir.mkdir(parents=True, exist_ok=True)

        notation_episode_prefix = f"-{n_prefix}..." if n_prefix > 0 else None
        notation_episode_suffix = None
        if suffix_pad > 0:
            notation_episode_suffix = f"...{T - 1}+{suffix_pad}"

        for chunk_i in tqdm(
            range(0, len(virtual_starts), batch_size),
            desc=f"windows {stem}",
            leave=False,
        ):
            chunk_v = virtual_starts[chunk_i : chunk_i + batch_size]
            B = len(chunk_v)
            stats = [_window_decomposition(v, W, n_prefix, T) for v in chunk_v]

            denoise_arg: float | None | torch.Tensor
            if args.noise_mode == "fixed":
                denoise_arg = float(args.denoise_level)
            elif args.noise_mode == "scheduler":
                denoise_arg = None
            else:
                levels = [rng.uniform(args.denoise_min, args.denoise_max) for _ in range(B)]
                denoise_arg = torch.tensor(levels, device=device, dtype=torch.float32)

            if args.skip_existing and args.noise_mode != "scheduler":
                candidate_paths: list[Path] = []
                for j in range(B):
                    head, tail, rl, rh = stats[j]
                    if isinstance(denoise_arg, torch.Tensor):
                        tag_pre = _noise_tag_denoise_level(float(denoise_arg[j].item()))
                    else:
                        assert denoise_arg is not None
                        tag_pre = _noise_tag_denoise_level(float(denoise_arg))
                    candidate_paths.append(ep_dir / _window_basename(head, rl, rh, tail, tag_pre))
                if all(p.is_file() for p in candidate_paths):
                    continue

            frames_list: list[torch.Tensor] = []
            for v in chunk_v:
                decord_idxs = [_virtual_to_decord_idx(v + j, n_prefix, T) for j in range(W)]
                batch = vr.get_batch(decord_idxs)
                frames_list.append(_decord_batch_to_tensor(batch, device))
            frames = torch.cat(frames_list, dim=0)

            result = adapter.maybe_extract_intermediate_features(
                frames,
                denoise_level=denoise_arg,
                prompt=prompt,
                return_metadata=True,
            )
            if result is None:
                print(f"warning: None features for {stem} batch starting v={chunk_v[0]}", file=sys.stderr)
                continue
            feats, meta = result

            for j, v in enumerate(chunk_v):
                head, tail, rl, rh = stats[j]
                noise_tag = _noise_tag_from_meta(meta, j)
                out_path = ep_dir / _window_basename(head, rl, rh, tail, noise_tag)
                if args.skip_existing and args.noise_mode == "scheduler" and out_path.is_file():
                    continue

                feat_j = feats[j].detach().cpu()
                dl_one = _payload_denoise_level(meta, j)

                payload = {
                    "features": feat_j,
                    "prompt": meta["prompt"],
                    "denoise_level": dl_one,
                    "sigma_b": meta["sigma_b"][j : j + 1].clone(),
                    "virtual_start": v,
                    "virtual_end_exclusive": v + W,
                    "prefix_pad_episode": n_prefix,
                    "suffix_pad_episode": suffix_pad,
                    "video_num_frames": T,
                    "num_conditional_frames": n_prefix,
                    "window_head_pad": head,
                    "window_tail_pad": tail,
                    "real_start": rl,
                    "real_end_exclusive": rh,
                    "notation_episode_prefix": notation_episode_prefix,
                    "notation_episode_suffix": notation_episode_suffix,
                    "start_frame": v,
                    "end_frame": v + W,
                    "episode_stem": stem,
                    "video_path": str(video_path.resolve()),
                    "caption_path": str(caption_path.resolve()) if caption_path.is_file() else None,
                    "noise_mode": args.noise_mode,
                    "window_frames": W,
                    "batch_size_effective": B,
                    "cosmos_intermediate_pool": args.pool,
                    "cosmos_block_index": args.block,
                    "noise_tag": noise_tag,
                    "output_filename": out_path.name,
                }
                torch.save(payload, out_path)

    print(f"Done. Outputs under {out_root}")


if __name__ == "__main__":
    main()
