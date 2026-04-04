from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from libero_future_policy.train.trainer import Trainer
from libero_future_policy.utils.config import load_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/experiment/oracle_future.yaml",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if args.max_steps is not None:
        config.train.max_steps = args.max_steps

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
