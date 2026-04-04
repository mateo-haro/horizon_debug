from __future__ import annotations


class StepLogger:
    def __init__(self, run_name: str) -> None:
        self.run_name = run_name

    def log(self, step: int, metrics: dict[str, float]) -> None:
        metric_str = " ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))
        print(f"[{self.run_name}] step={step:05d} {metric_str}")
