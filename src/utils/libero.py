from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path


LIBERO_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
    "libero_100",
)

LIBERO_DATASET_FLAVORS = ("image", "state")


def validate_suite_name(suite: str) -> None:
    if suite not in LIBERO_SUITES:
        raise ValueError(f"Unsupported LIBERO suite: {suite}. Expected one of {LIBERO_SUITES}.")


def validate_dataset_flavor(dataset_flavor: str) -> None:
    if dataset_flavor not in LIBERO_DATASET_FLAVORS:
        raise ValueError(
            f"Unsupported dataset flavor: {dataset_flavor}. Expected one of {LIBERO_DATASET_FLAVORS}."
        )


def dataset_repo_id_for_suite(suite: str, dataset_flavor: str = "image") -> str:
    validate_suite_name(suite)
    validate_dataset_flavor(dataset_flavor)
    if dataset_flavor == "image":
        return f"lerobot/{suite}_image"
    return f"lerobot/{suite}"


def discover_local_libero_root(start: str | Path) -> Path | None:
    start_path = Path(start).resolve()
    search_roots = [start_path] + list(start_path.parents)
    for root in search_roots:
        direct = root / "LIBERO"
        if (direct / "libero" / "libero" / "benchmark" / "__init__.py").exists():
            return direct

        lowercase = root / "libero"
        if (lowercase / "libero" / "benchmark" / "__init__.py").exists():
            return lowercase
    return None


def _maybe_import_libero_benchmark(libero_root: str | Path | None = None):
    try:
        return importlib.import_module("libero.libero.benchmark")
    except Exception:
        pass

    candidate = None
    if libero_root is not None:
        candidate = Path(libero_root).resolve() / "libero"
    else:
        discovered = discover_local_libero_root(Path.cwd())
        if discovered is not None:
            candidate = discovered / "libero"

    if candidate is None:
        return None

    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)
    try:
        return importlib.import_module("libero.libero.benchmark")
    except Exception:
        return None


@lru_cache(maxsize=16)
def suite_task_texts(suite: str, libero_root: str | Path | None = None) -> tuple[str, ...]:
    validate_suite_name(suite)
    benchmark_module = _maybe_import_libero_benchmark(libero_root)
    if benchmark_module is None:
        return tuple()

    benchmark_cls = benchmark_module.get_benchmark(suite)
    benchmark = benchmark_cls()
    return tuple(str(task.language) for task in benchmark.tasks)


def task_texts_from_indices(
    suite: str,
    task_indices: list[int] | tuple[int, ...],
    libero_root: str | Path | None = None,
) -> list[str]:
    texts = suite_task_texts(suite, libero_root)
    if not texts:
        return ["" for _ in task_indices]
    return [texts[index] if 0 <= index < len(texts) else "" for index in task_indices]
