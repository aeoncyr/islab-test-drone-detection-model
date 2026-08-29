"""
src/utils/logger.py

Unified metric logger: prints to console and optionally logs to W&B.
"""

from typing import Any, Dict, Optional


class MetricLogger:
    """
    Lightweight wrapper for logging training metrics.

    Handles both console output (always) and Weights & Biases (optional).
    If W&B is unavailable or disabled via config, it silently degrades
    to console-only logging — training is never blocked.

    Args:
        project  : W&B project name.
        run_name : W&B run name (typically the model config name).
        cfg      : Full config dict (logged as W&B hyperparameters).
        enabled  : Whether to attempt W&B logging. Default: True.
    """

    def __init__(
        self,
        project: str,
        run_name: str,
        cfg: Dict[str, Any],
        enabled: bool = True,
    ) -> None:
        self._wandb = None
        self._enabled = enabled

        if enabled:
            try:
                import wandb
                wandb.init(project=project, name=run_name, config=cfg)
                self._wandb = wandb
                print(f"  [W&B] Run initialized: {project}/{run_name}")
            except Exception as e:
                print(f"  [W&B] Disabled — {e}")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log a dict of metrics.

        Args:
            metrics: Key-value pairs to log (e.g., {"train/loss": 0.42}).
            step   : Global step/epoch number. Optional.
        """
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        """Close the W&B run gracefully."""
        if self._wandb is not None:
            self._wandb.finish()
