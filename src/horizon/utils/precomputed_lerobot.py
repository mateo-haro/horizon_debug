"""Joining offline Cosmos ``.pt`` windows with LeRobot LIBERO samples.

Training with ``use_precomputed_cosmos_latents=True`` requires each batch to include:

- ``horizon.precomputed_curr_vis``: float tensor, raw or mean-pooled Cosmos latents for current frames.
- ``horizon.precomputed_future_vis``: same for future frames.

Shape contract: rank-3 ``[batch, time, D_latent]`` after any flattening you apply in ``datasets.map``;
:class:`horizon.models.visual_latent_projection.VisualLatentProjection` maps
``cosmos_encode_token_dim`` / ``cosmos_hidden_token_dim`` → ``cosmos_feature_dim``.

**Tier A (recommended):** Hugging Face ``datasets`` + ``.map``:

1. Run :mod:`scripts.build_cosmos_intermediate_index` (or :func:`scan_intermediates_root`) on your
   ``--out-dir`` from extraction.
2. For each dataset row, resolve the episode id / frame index to a window ``.pt`` path (e.g. overlap
   ``real_lo <= frame < real_hi`` from the filename).
3. ``torch.load`` the file, take ``payload["features"]``, reshape to ``[1, T, D]`` if needed, stack
   for current vs future according to your alignment rule.
4. Return new columns ``horizon.precomputed_curr_vis`` and ``horizon.precomputed_future_vis``.

5. Point ``lerobot-train`` at the mapped dataset (local path or new Hub revision) and launch via
   :mod:`scripts.train_horizon_libero_precomputed`.

This module stays dependency-light; it does not import ``lerobot`` at import time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def scan_intermediates_root(intermediates_root: str | Path) -> Iterator[dict[str, Any]]:
    """Yield one dict per ``*.pt`` file (path + episode stem) without loading tensors."""
    root = Path(intermediates_root).resolve()
    for ep_dir in sorted(root.iterdir()):
        if not ep_dir.is_dir():
            continue
        episode_stem = ep_dir.name
        for pt in sorted(ep_dir.glob("*.pt")):
            yield {"episode_stem": episode_stem, "path": str(pt.resolve())}


def write_jsonl_index(intermediates_root: str | Path, out_jsonl: str | Path) -> int:
    """Write a JSONL index compatible with :func:`scan_intermediates_root` for manual joins."""
    out_path = Path(out_jsonl).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in scan_intermediates_root(intermediates_root):
            f.write(json.dumps(row) + "\n")
            n += 1
    return n
