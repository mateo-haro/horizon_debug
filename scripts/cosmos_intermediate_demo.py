"""Manual demo: dummy video -> CosmosAdapter -> intermediate DiT features after block k.

Prerequisites (otherwise the script skips or exits non-zero with --require-success):

- NVIDIA GPU with CUDA available
- ``cosmos-predict2`` (and imaginaire) installed, e.g.::

    uv pip install -U \"cosmos-predict2[cu126]\"
        --extra-index-url https://nvidia-cosmos.github.io/cosmos-dependencies/cu126_torch260/simple

- Checkpoint reachable via ``get_cosmos_predict2_video2world_checkpoint`` for your
  ``--model-size`` / ``--resolution`` / ``--fps`` / ``--aspect-ratio`` / ``--natten``,
  or pass ``--dit-path`` to a local weights file.

Run from repo root::

    python scripts/cosmos_intermediate_demo.py

First-time weights (Video2World + T5) download under the checkpoints **root** chosen by
``--checkpoints-root``, or env ``COSMOS_CHECKPOINTS_DIR``, or ``~/.cache/horizon/cosmos_checkpoints``.

By default, missing prerequisites print a skip message to stderr and exit 0.
Use ``--require-success`` to exit 1 when feature extraction does not run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from libero_future_policy.backbones.cosmos_adapter import CosmosAdapter, CosmosAdapterConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo: extract Cosmos DiT intermediate features from dummy video.")
    parser.add_argument("--block", type=int, default=0, help="DiT block index k (clamped by adapter).")
    parser.add_argument("--feature-dim", type=int, default=128, help="Adapter feature_dim and optional projection target.")
    parser.add_argument(
        "--denoise-level",
        type=float,
        default=0.5,
        help="Sigma interpolant in [0, 1]. Ignored if --random-sigma.",
    )
    parser.add_argument(
        "--random-sigma",
        action="store_true",
        help="Sample sigma via the training scheduler instead of a fixed denoise level.",
    )
    parser.add_argument("--prompt", type=str, default="A robot arm manipulates objects on a table.", help="Text prompt.")
    parser.add_argument("--dit-path", type=str, default=None, help="Override path to DiT checkpoint.")
    parser.add_argument(
        "--checkpoints-root",
        type=str,
        default=None,
        help=(
            "Directory whose children include nvidia/... and google-t5/... "
            "(default: COSMOS_CHECKPOINTS_DIR or ~/.cache/horizon/cosmos_checkpoints)."
        ),
    )
    parser.add_argument("--pool", type=str, choices=("mean", "none"), default="mean", help="cosmos_intermediate_pool.")
    parser.add_argument("--model-size", type=str, default="2B", help="Cosmos Video2World model size.")
    parser.add_argument("--resolution", type=str, default="720", help="e.g. 720 or 480.")
    parser.add_argument("--fps", type=int, default=16, choices=(10, 16), help="Pipeline fps.")
    parser.add_argument("--aspect-ratio", type=str, default="16:9", help="Must exist in VIDEO_RES_SIZE_INFO for resolution.")
    parser.add_argument("--natten", action="store_true", help="Use NATTEN pipeline variant.")
    parser.add_argument("--time-frames", type=int, default=8, help="Dummy temporal length before adapter resizes.")
    parser.add_argument("--height", type=int, default=64, help="Dummy H before adapter resizes.")
    parser.add_argument("--width", type=int, default=64, help="Dummy W before adapter resizes.")
    parser.add_argument(
        "--require-success",
        action="store_true",
        help="Exit 1 if CUDA, pipeline load, or extraction fails (default: exit 0 when skipped).",
    )
    args = parser.parse_args()

    def skip(msg: str, code: int = 0) -> None:
        print(msg, file=sys.stderr)
        raise SystemExit(code if not args.require_success else 1)

    if not torch.cuda.is_available():
        skip("Skip: CUDA is not available (torch.cuda.is_available() is False). A GPU and CUDA build of PyTorch are required.")

    frames = torch.rand(
        1,
        3,
        args.time_frames,
        args.height,
        args.width,
        device="cuda",
        dtype=torch.float32,
    ).clamp(0.0, 1.0)

    try:
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
                cosmos_num_conditional_frames=1,
                cosmos_intermediate_pool=args.pool,
            )
        )
    except Exception as exc:
        skip(f"Skip: CosmosAdapter failed during init ({type(exc).__name__}: {exc}).")

    if getattr(adapter, "_cosmos_pipe", None) is None:
        reason = getattr(adapter, "cosmos_pipeline_load_error", None)
        if reason:
            skip(f"Skip: Cosmos pipeline did not load.\nReason: {reason}")
        skip(
            "Skip: Cosmos pipeline did not load and no detailed reason was recorded. "
            "If use_external_cosmos is True, install cosmos-predict2, place checkpoints under "
            "COSMOS_CHECKPOINTS_DIR or --checkpoints-root, or pass --dit-path.",
        )

    denoise_level = None if args.random_sigma else args.denoise_level
    features = adapter.maybe_extract_intermediate_features(
        frames,
        denoise_level=denoise_level,
        prompt=args.prompt,
    )
    if features is None:
        skip("Skip: maybe_extract_intermediate_features returned None (unexpected after pipeline load).")

    print(f"features shape: {tuple(features.shape)}")
    print(f"features dtype: {features.dtype}")
    print(f"block index: {args.block}")
    print(f"pool: {args.pool}")


if __name__ == "__main__":
    main()
