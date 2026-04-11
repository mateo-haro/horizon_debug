from __future__ import annotations

from pathlib import Path

from utils.libero import dataset_repo_id_for_suite, discover_local_libero_root, task_texts_from_indices


def test_dataset_repo_id_mapping() -> None:
    assert dataset_repo_id_for_suite("libero_spatial", "image") == "lerobot/libero_spatial_image"
    assert dataset_repo_id_for_suite("libero_10", "state") == "lerobot/libero_10"


def test_invalid_suite_raises() -> None:
    try:
        dataset_repo_id_for_suite("not_a_suite", "image")
    except ValueError:
        return
    raise AssertionError("Expected ValueError for an unsupported suite.")


def test_discover_local_libero_root(tmp_path: Path) -> None:
    libero_root = tmp_path / "LIBERO"
    benchmark_dir = libero_root / "libero" / "libero" / "benchmark"
    benchmark_dir.mkdir(parents=True)
    (benchmark_dir / "__init__.py").write_text("", encoding="utf-8")

    child = libero_root / "subdir" / "work"
    child.mkdir(parents=True)
    assert discover_local_libero_root(child) == libero_root


def test_task_texts_without_libero_import_are_safe() -> None:
    texts = task_texts_from_indices("libero_spatial", [0, 1], libero_root="/tmp/does-not-exist")
    assert texts == ["", ""]
