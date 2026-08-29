"""
src/utils/checkpoint.py

Checkpoint management utilities for saving and loading model states.
"""

import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class CheckpointManager:
    """
    Manages saving and loading of model checkpoints.

    Tracks the best validation metric and always saves the best checkpoint.
    Also saves periodic checkpoints at configurable intervals.

    Args:
        save_dir     : Directory to write checkpoint files.
        save_interval: Save a numbered checkpoint every N epochs. Default: 10.
        metric_name  : Name of the metric to maximize (e.g. "mAP50"). Default: "mAP50".
    """

    def __init__(
        self,
        save_dir: str,
        save_interval: int = 10,
        metric_name: str = "mAP50",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval
        self.metric_name = metric_name
        self.best_metric = -1.0
        self.best_epoch = 0

    @property
    def best_score(self) -> float:
        """Alias for best_metric."""
        return self.best_metric

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        metrics: Dict[str, float],
    ) -> None:
        """
        Save checkpoint if epoch is a save interval or a new best is found.

        Args:
            epoch    : Current epoch number (1-indexed).
            model    : Model to save (handles DataParallel wrapper).
            optimizer: Optimizer state to save.
            scheduler: LR scheduler state to save.
            metrics  : Dict of validation metrics (must include `metric_name`).
        """
        state = {
            "epoch": epoch,
            "model_state_dict": self._unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
        }

        # Save periodic checkpoint
        if epoch % self.save_interval == 0:
            ckpt_path = self.save_dir / f"epoch_{epoch:04d}.pth"
            try:
                torch.save(state, ckpt_path)
                print(f"  [Checkpoint] Saved → {ckpt_path}")
            except OSError as e:
                warnings.warn(
                    f"[Checkpoint] Failed to save periodic checkpoint: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Save best checkpoint
        current_metric = metrics.get(self.metric_name, -1.0)
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.best_epoch = epoch
            best_path = self.save_dir / "best.pth"
            try:
                torch.save(state, best_path)
                print(f"  [Checkpoint] New best {self.metric_name}={current_metric:.4f} → {best_path}")
            except OSError as e:
                warnings.warn(
                    f"[Checkpoint] Failed to save best checkpoint: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Always overwrite latest
        latest_path = self.save_dir / "latest.pth"
        try:
            torch.save(state, latest_path)
        except OSError as e:
            warnings.warn(
                f"[Checkpoint] Failed to save latest checkpoint: {e}",
                RuntimeWarning,
                stacklevel=2,
            )

    @staticmethod
    def load(
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint and restore model (and optionally optimizer/scheduler) states.

        Args:
            checkpoint_path: Path to the .pth checkpoint file.
            model          : Model to load weights into.
            optimizer      : Optional optimizer to restore. Default: None.
            scheduler      : Optional scheduler to restore. Default: None.
            device         : Target device. Default: auto-detected.

        Returns:
            The checkpoint dict (contains "epoch" and "metrics").
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_path}. "
                f"Ensure training completed and the path is correct."
            )

        try:
            state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint '{checkpoint_path}': {e}. "
                f"The file may be corrupted."
            ) from e

        unwrapped = CheckpointManager._unwrap_model(model)
        unwrapped.load_state_dict(state["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])

        print(f"  [Checkpoint] Loaded epoch {state.get('epoch', '?')} from {checkpoint_path}")
        return state

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        """Unwrap DataParallel / DistributedDataParallel to get the raw module."""
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return model.module
        return model
