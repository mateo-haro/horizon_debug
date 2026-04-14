"""Parse filenames and load payloads from ``extract_libero_cosmos_intermediates.py`` outputs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# window_h<head>_r<real_lo>_<real_hi>_t<tail>_<dl|sg><encoded>.pt
_WINDOW_RE = re.compile(
    r"^window_h(\d+)_r(-?\d+)_(-?\d+)_t(\d+)_(dl|sg)(.+)\.pt$",
    re.IGNORECASE,
)


def decode_noise_tag_component(s: str) -> float:
    """Invert ``_noise_tag_denoise_level`` / ``_noise_tag_sigma`` filesystem encoding."""
    t = s.replace("p", ".").replace("m", "-")
    return float(t)


def parse_intermediate_filename(name: str) -> dict[str, Any]:
    """Parse ``window_h*_r*_t*_*.pt`` stem or filename."""
    stem = Path(name).name
    m = _WINDOW_RE.match(stem)
    if not m:
        raise ValueError(f"Unrecognized intermediate filename: {name!r}")
    head = int(m.group(1))
    real_lo = int(m.group(2))
    real_hi = int(m.group(3))
    tail = int(m.group(4))
    tag_kind = m.group(5).lower()
    tag_body = m.group(6)
    value = decode_noise_tag_component(tag_body)
    out: dict[str, Any] = {
        "window_head_pad": head,
        "real_lo": real_lo,
        "real_hi": real_hi,
        "window_tail_pad": tail,
        "noise_tag_kind": tag_kind,
        "noise_tag_body": tag_body,
    }
    if tag_kind == "dl":
        out["denoise_level"] = value
    else:
        out["sigma_value"] = value
    return out


def load_intermediate_payload(path: str | Path, *, map_location: str | Any = "cpu") -> dict[str, Any]:
    """``torch.load`` a saved window dict and attach ``parse_intermediate_filename`` metadata."""
    import torch

    path = Path(path)
    try:
        raw = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        raw = torch.load(path, map_location=map_location)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(raw)}")
    try:
        meta = parse_intermediate_filename(path.name)
    except ValueError:
        meta = {}
    out = dict(raw)
    out["filename_parsed"] = meta
    out["path"] = str(path.resolve())
    return out
