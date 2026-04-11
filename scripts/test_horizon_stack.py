from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


class Reporter:
    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def add(self, name: str, status: str, detail: str) -> None:
        self.results.append(CheckResult(name=name, status=status, detail=detail))
        print(f"[{status}] {name}: {detail}")

    def pass_(self, name: str, detail: str) -> None:
        self.add(name, "PASS", detail)

    def warn(self, name: str, detail: str) -> None:
        self.add(name, "WARN", detail)

    def fail(self, name: str, detail: str) -> None:
        self.add(name, "FAIL", detail)

    def summary(self) -> int:
        print("\n=== Summary ===")
        counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
        for item in self.results:
            counts[item.status] += 1
        print(f"PASS={counts['PASS']} WARN={counts['WARN']} FAIL={counts['FAIL']}")
        for item in self.results:
            print(f"- {item.status:4s} {item.name}: {item.detail}")
        return 1 if counts["FAIL"] > 0 else 0


def _exception_detail(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _repo_on_path(repo_path: Path) -> None:
    if repo_path.exists() and str(repo_path.resolve()) not in sys.path:
        sys.path.insert(0, str(repo_path.resolve()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Horizon + Cosmos + LeRobot stack on AWS.")
    parser.add_argument("--repo-path", type=Path, default=ROOT / "cosmos-predict2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--hidden-layer", type=int, default=3)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--noise-level", type=float, default=0.15)
    parser.add_argument("--skip-real-cosmos", action="store_true")
    parser.add_argument("--skip-policy", action="store_true")
    parser.add_argument("--cosmos-lora-checkpoint", type=str, default=None)
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=None,
        help=(
            "Absolute path to the Cosmos checkpoints *root* (directory that contains "
            "`nvidia/Cosmos-Predict2-.../` and `google-t5/`). Sets COSMOS_CHECKPOINTS_DIR and "
            "COSMOS_PREDICT2_ARGS so tokenizer/DiT resolve without relying on cwd. "
            "Example: /data/cosmos-predict2/checkpoints"
        ),
    )
    return parser.parse_args()


def check_system(reporter: Reporter) -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {}
    info["python"] = sys.version.split()[0]
    info["platform"] = platform.platform()
    info["cwd"] = str(Path.cwd())
    info["torch"] = torch.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    info["cuda_version"] = torch.version.cuda

    reporter.pass_(
        "system",
        f"python={info['python']} torch={info['torch']} cuda_available={info['cuda_available']} "
        f"cuda_devices={info['cuda_device_count']}",
    )

    if torch.cuda.is_available():
        try:
            names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            reporter.pass_("gpu_names", ", ".join(names))
        except Exception as exc:
            reporter.warn("gpu_names", _exception_detail(exc))
    else:
        reporter.warn("gpu_names", "CUDA is not available; real Cosmos runtime tests will be skipped or fail.")

    return info


def apply_cosmos_checkpoints_root(checkpoints_root: Path) -> None:
    """Must run before importing cosmos_predict2 / imaginaire checkpoint paths."""
    root = checkpoints_root.expanduser().resolve()
    os.environ["COSMOS_CHECKPOINTS_DIR"] = str(root)
    from horizon.backbones.cosmos_adapter import _apply_cosmos_predict2_checkpoints_root

    _apply_cosmos_predict2_checkpoints_root(root)


def check_cosmos_checkpoints_layout(reporter: Reporter, checkpoints_root: Path | None) -> None:
    if checkpoints_root is None:
        return
    root = checkpoints_root.expanduser().resolve()
    if not root.is_dir():
        reporter.warn("cosmos_checkpoints", f"Not a directory: {root}")
        return
    tok = root / "nvidia" / "Cosmos-Predict2-2B-Video2World" / "tokenizer" / "tokenizer.pth"
    if tok.is_file():
        reporter.pass_("cosmos_checkpoints", str(root))
    else:
        reporter.warn(
            "cosmos_checkpoints",
            f"No tokenizer at {tok} — DiT may still fail to load until weights match this layout.",
        )


def check_repo_layout(reporter: Reporter, repo_path: Path) -> None:
    if ROOT.exists():
        reporter.pass_("repo_root", str(ROOT))
    else:
        reporter.fail("repo_root", f"Missing repo root: {ROOT}")

    if repo_path.exists():
        reporter.pass_("cosmos_repo", str(repo_path.resolve()))
    else:
        reporter.fail("cosmos_repo", f"Missing Cosmos repo at {repo_path}")

    libero_doc = repo_path / "scripts" / "libero.md"
    if libero_doc.exists():
        reporter.pass_("cosmos_libero_doc", str(libero_doc))
    else:
        reporter.warn("cosmos_libero_doc", f"Missing {libero_doc}")


def check_imports(reporter: Reporter, repo_path: Path) -> None:
    _repo_on_path(repo_path)

    modules = [
        "yaml",
        "torch",
        "utils.config",
        "utils.lerobot",
        "horizon.configuration_horizon_dit",
        "horizon.modeling_horizon_dit",
        "horizon.backbones.cosmos_adapter",
        "horizon.future_sources.sources",
        "lerobot",
        "libero",
        "omegaconf",
        "hydra",
        "einops",
        "megatron",
        "imaginaire",
        "peft",
        "cosmos_predict2",
    ]
    for module_name in modules:
        try:
            __import__(module_name)
            reporter.pass_(f"import:{module_name}", "ok")
        except Exception as exc:
            reporter.warn(f"import:{module_name}", _exception_detail(exc))


def check_launcher_dry_run(reporter: Reporter) -> None:
    try:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "train_horizon_libero.py"), "--dry-run"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        command = proc.stdout.strip().splitlines()[-1]
        reporter.pass_("launcher_dry_run", command)
    except Exception as exc:
        reporter.fail("launcher_dry_run", _exception_detail(exc))


def check_real_cosmos_adapter(reporter: Reporter, args: argparse.Namespace, system_info: dict[str, Any]) -> bool:
    import torch
    from horizon.backbones.cosmos_adapter import CosmosAdapter

    if args.skip_real_cosmos:
        reporter.warn("real_cosmos_adapter", "Skipped by flag.")
        return False

    if args.device.startswith("cuda") and not system_info["cuda_available"]:
        reporter.warn("real_cosmos_adapter", "CUDA requested but not available.")
        return False

    try:
        adapter = CosmosAdapter(
            feature_dim=args.feature_dim,
            hidden_layer_index=args.hidden_layer,
            noise_level=args.noise_level,
            num_hidden_layers=max(args.hidden_layer + 1, 8),
            device=args.device,
            use_external_runtime=True,
            repo_path=str(args.repo_path),
            model_size="2B",
            resolution="480",
            fps=10,
            lora_checkpoint=args.cosmos_lora_checkpoint,
        )
        adapter._ensure_runtime()
        if adapter.runtime is None:
            detail = "runtime did not initialize"
            if adapter.runtime_error is not None:
                detail = f"{detail}: {_exception_detail(adapter.runtime_error)}"
            reporter.warn("real_cosmos_adapter", detail)
            return False

        frames = torch.randn(1, args.frames, 3, args.height, args.width, device=args.device)
        with torch.inference_mode():
            encoded = adapter.encode(frames)

        reporter.pass_(
            "real_cosmos_encode",
            f"encoded={tuple(encoded.shape)} latent_frames={adapter.feature_num_frames(args.frames)}",
        )

        with torch.inference_mode():
            state = adapter.denoise_to_tau(
                frames,
                tau=args.tau,
                hidden_layer_index=args.hidden_layer,
                noise_level=args.noise_level,
                texts=["open the drawer"],
            )
            hidden = adapter.get_nth_hidden_layer(state, hidden_layer_index=args.hidden_layer)
        reporter.pass_(
            "real_cosmos_denoise",
            f"hidden_layers={len(state.hidden_states)} hidden={tuple(hidden.shape)} sigma={state.sigma}",
        )
        return True
    except torch.cuda.OutOfMemoryError as exc:
        reporter.fail("real_cosmos_adapter", f"OOM: {exc}")
        return False
    except Exception as exc:
        reporter.fail("real_cosmos_adapter", _exception_detail(exc))
        tb = traceback.format_exc(limit=6)
        print(tb)
        return False


def check_policy(reporter: Reporter, args: argparse.Namespace, use_real_cosmos: bool) -> None:
    import torch
    from horizon.configuration_horizon_dit import HorizonDiTConfig
    from horizon.modeling_horizon_dit import HorizonDiTPolicy

    if args.skip_policy:
        reporter.warn("policy", "Skipped by flag.")
        return

    policy_device = args.device if use_real_cosmos else "cpu"
    image_h = args.height if use_real_cosmos else 64
    image_w = args.width if use_real_cosmos else 64

    try:
        cfg = HorizonDiTConfig(
            device=policy_device,
            current_obs_steps=2,
            future_obs_steps=2,
            chunk_size=6,
            n_action_steps=3,
            model_dim=128,
            depth=2,
            num_heads=4,
            vis_feature_dim=args.feature_dim,
            cosmos_feature_dim=args.feature_dim,
            cosmos_num_hidden_layers=max(args.hidden_layer + 1, 8),
            cosmos_hidden_layer=args.hidden_layer,
            cosmos_noise_level=args.noise_level,
            cosmos_tau=args.tau,
            future_source="oracle_hidden",
            use_task_text=True,
            use_proprio=True,
            libero_suite=None,
            cosmos_use_external_runtime=use_real_cosmos,
            cosmos_repo_path=str(args.repo_path),
            cosmos_model_size="2B",
            cosmos_resolution="480",
            cosmos_fps=10,
            cosmos_lora_checkpoint=args.cosmos_lora_checkpoint,
        )
        policy = HorizonDiTPolicy(cfg).to(policy_device)
        batch = {
            "observation.images.image": torch.randn(1, 4, 3, image_h, image_w, device=policy_device),
            "observation.state": torch.randn(1, 4, 8, device=policy_device),
            "action": torch.randn(1, 6, 7, device=policy_device),
            "task_text": ["open the drawer"],
        }
        loss, info = policy.forward(batch)
        reporter.pass_("policy_forward", f"loss={float(loss.item()):.6f} info_keys={sorted(info.keys())}")
    except torch.cuda.OutOfMemoryError as exc:
        reporter.fail("policy_forward", f"OOM: {exc}")
    except Exception as exc:
        reporter.fail("policy_forward", _exception_detail(exc))


def main() -> int:
    args = parse_args()
    reporter = Reporter()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.checkpoints is not None:
        apply_cosmos_checkpoints_root(args.checkpoints)
    check_repo_layout(reporter, args.repo_path)
    check_cosmos_checkpoints_layout(reporter, args.checkpoints)
    system_info = check_system(reporter)
    check_imports(reporter, args.repo_path)
    check_launcher_dry_run(reporter)
    use_real_cosmos = check_real_cosmos_adapter(reporter, args, system_info)
    check_policy(reporter, args, use_real_cosmos=use_real_cosmos)
    return reporter.summary()


if __name__ == "__main__":
    raise SystemExit(main())
