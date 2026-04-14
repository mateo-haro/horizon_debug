"""Scan ``extract_libero_cosmos_intermediates.py`` output tree and emit a JSONL index.

Each line: JSON with ``episode_stem``, ``path``, ``filename_parsed`` (from
``horizon.utils.cosmos_intermediate_io.parse_intermediate_filename``), and payload keys
``real_start``, ``virtual_start`` if present after ``torch.load`` (optional, slow).

Example::

  python scripts/build_cosmos_intermediate_index.py \\
    --intermediates-root outputs/cosmos_intermediates \\
    --out-jsonl outputs/cosmos_intermediates/index.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from horizon.utils.cosmos_intermediate_io import load_intermediate_payload, parse_intermediate_filename


def main() -> None:
    p = argparse.ArgumentParser(description="Build JSONL index of Cosmos intermediate .pt files.")
    p.add_argument("--intermediates-root", type=Path, required=True, help="Root containing episode subdirs.")
    p.add_argument("--out-jsonl", type=Path, required=True)
    p.add_argument(
        "--load-payload",
        action="store_true",
        help="torch.load each file to attach real_start, virtual_start, etc. (slow).",
    )
    args = p.parse_args()

    root = args.intermediates_root.resolve()
    out = args.out_jsonl.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out.open("w", encoding="utf-8") as f:
        for ep_dir in sorted(root.iterdir()):
            if not ep_dir.is_dir():
                continue
            episode_stem = ep_dir.name
            for pt in sorted(ep_dir.glob("*.pt")):
                row: dict = {
                    "episode_stem": episode_stem,
                    "path": str(pt.resolve()),
                }
                try:
                    row["filename_parsed"] = parse_intermediate_filename(pt.name)
                except ValueError as exc:
                    row["filename_parse_error"] = str(exc)
                if args.load_payload:
                    try:
                        payload = load_intermediate_payload(pt)
                        for k in (
                            "real_start",
                            "real_end_exclusive",
                            "virtual_start",
                            "virtual_end_exclusive",
                            "video_num_frames",
                            "window_frames",
                        ):
                            if k in payload:
                                row[k] = payload[k]
                    except Exception as exc:
                        row["payload_error"] = f"{type(exc).__name__}: {exc}"
                f.write(json.dumps(row, default=str) + "\n")
                n += 1

    print(f"Wrote {n} rows to {out}")


if __name__ == "__main__":
    main()
