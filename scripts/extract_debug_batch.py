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
        default=ROOT / "configs/experiment/mixed_future.yaml",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    trainer = Trainer(config)
    batch = next(trainer.iter_batches())
    debug = trainer.extract_debug_batch(batch)

    for key, value in debug.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
