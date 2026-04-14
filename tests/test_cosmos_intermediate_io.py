from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "horizon" / "utils" / "cosmos_intermediate_io.py"


def _load_io():
    spec = importlib.util.spec_from_file_location("cosmos_intermediate_io", _SRC)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cio = _load_io()


def test_decode_noise_tag_component() -> None:
    assert cio.decode_noise_tag_component("0p500000") == pytest.approx(0.5)
    assert cio.decode_noise_tag_component("m0p100000") == pytest.approx(-0.1)


def test_parse_intermediate_filename_roundtrip() -> None:
    name = "window_h005_r10_20_t003_dl0p500000.pt"
    meta = cio.parse_intermediate_filename(name)
    assert meta["window_head_pad"] == 5
    assert meta["real_lo"] == 10
    assert meta["real_hi"] == 20
    assert meta["window_tail_pad"] == 3
    assert meta["noise_tag_kind"] == "dl"
    assert meta["denoise_level"] == pytest.approx(0.5)


def test_parse_invalid_raises() -> None:
    with pytest.raises(ValueError):
        cio.parse_intermediate_filename("not_a_window.pt")
