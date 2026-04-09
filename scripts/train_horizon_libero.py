from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from horizon.config import HorizonLaunchConfig, load_horizon_launch_config
from horizon.lerobot import build_train_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch LeRobot training for the HorizonDiT LIBERO policy.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/train/libero_horizon_oracle_hidden.yaml",
    )
    parser.add_argument("--suite", type=str, default=None)
    parser.add_argument("--dataset-repo-id", type=str, default=None)
    parser.add_argument("--dataset-flavor", type=str, choices=["image", "state"], default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("passthrough", nargs=argparse.REMAINDER)
    return parser.parse_args()


def apply_overrides(config: HorizonLaunchConfig, args: argparse.Namespace) -> HorizonLaunchConfig:
    if args.suite is not None:
        config.train["suite"] = args.suite
    if args.dataset_repo_id is not None:
        config.train["dataset_repo_id"] = args.dataset_repo_id
    if args.dataset_flavor is not None:
        config.train["dataset_flavor"] = args.dataset_flavor
    return config


def normalize_passthrough(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def maybe_refresh_default_names(
    original: HorizonLaunchConfig,
    updated: HorizonLaunchConfig,
    args: argparse.Namespace,
) -> HorizonLaunchConfig:
    default_suite = original.train.get("suite", "libero_spatial")
    default_policy_type = original.policy.get("type", "horizon_dit")
    new_suite = updated.train.get("suite", default_suite)
    new_policy_type = updated.policy.get("type", default_policy_type)

    old_suffix = f"{default_suite}_{default_policy_type}"
    new_suffix = f"{new_suite}_{new_policy_type}"

    output_dir = str(updated.train.get("output_dir", ""))
    if output_dir.endswith(old_suffix):
        updated.train["output_dir"] = output_dir[: -len(old_suffix)] + new_suffix

    job_name = str(updated.train.get("job_name", ""))
    if job_name == old_suffix:
        updated.train["job_name"] = new_suffix
    return updated


def main() -> None:
    args = parse_args()
    original = load_horizon_launch_config(args.config)
    config = apply_overrides(deepcopy(original), args)
    config = maybe_refresh_default_names(original, config, args)
    passthrough = normalize_passthrough(args.passthrough)
    command = build_train_command(config, passthrough)

    if args.print_config:
        print(config)
    print(shlex.join(command))

    if args.dry_run:
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
