"""
src/engine/trainer.py

Trainer — OOP training orchestrator for the VanillaDroneDetector.

Encapsulates the entire training lifecycle:
    1. Build model (with optional multi-GPU wrapping)
    2. Build optimizer, LR scheduler, AMP scaler
    3. Build DataLoaders (train + val)
    4. Train epoch → validate → checkpoint → log

Instantiated from a full config dict (loaded from YAML).
"""

import csv
import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.data.dataset import DroneYOLODataset
from src.data.transforms import build_transforms
from src.engine.evaluator import Evaluator
from src.engine.target_assigner import TargetAssigner
from src.losses.multi_task_losses import HybridMultiTaskLoss
from src.models.detector import VanillaDroneDetector
from src.utils.checkpoint import CheckpointManager
from src.utils.ema import ModelEMA
from src.utils.logger import MetricLogger


class Trainer:
    """
    Full training lifecycle manager supporting both Single-GPU, Multi-GPU DataParallel,
    and DistributedDataParallel (DDP via torchrun).

    Args:
        cfg: Nested dict loaded from a YAML config file. Expected keys:
             model, data, augmentation, training, loss, scheduler, wandb.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg

        # ── Distributed Process Configuration ──────────────────────
        self.is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))

        if self.is_distributed and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)

        self.device = self._setup_device()

        # ── Model ─────────────────────────────────────────────────
        self.model = self._build_model()

        # ── Model EMA ──────────────────────────────────────────────
        train_cfg = cfg["training"]
        self.use_ema = train_cfg.get("use_ema", False) or cfg.get("model", {}).get("use_ema", False)
        # Only instantiate EMA on master rank or main thread
        self.ema = ModelEMA(self.model) if self.use_ema else None
        if self.use_ema and self.rank == 0:
            print("[*] Model EMA enabled (decay=0.9999)")

        # ── Loss ──────────────────────────────────────────────────
        loss_cfg = cfg.get("loss", {})
        self.criterion = HybridMultiTaskLoss(
            num_classes=cfg["model"]["num_classes"],
            lambda_cls=loss_cfg.get("lambda_cls", 1.0),
            lambda_reg=loss_cfg.get("lambda_reg", 2.0),
            lambda_obj=loss_cfg.get("lambda_obj", 1.0),
            alpha_hybrid=loss_cfg.get("alpha_hybrid", 0.5),
            focal_alpha=loss_cfg.get("focal_alpha", 0.25),
            focal_gamma=loss_cfg.get("focal_gamma", 2.0),
            nwd_constant=loss_cfg.get("nwd_constant", 12.8),
        ).to(self.device)

        # ── Optimizer & Scheduler ──────────────────────────────────
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg.get("weight_decay", 1e-4),
        )

        self.warmup_epochs = train_cfg.get("warmup_epochs", 3)

        sched_cfg = cfg.get("scheduler", {})
        self.scheduler = self._build_scheduler(train_cfg["epochs"], sched_cfg)

        # AMP scaler (no-op on CPU)
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(train_cfg.get("amp", True) and self.device.type == "cuda")
        )

        # ── Target Assigner ────────────────────────────────────────
        data_cfg = cfg["data"]
        model_strides = getattr(self.model, "strides", None)
        if hasattr(self.model, "module"):
            model_strides = getattr(self.model.module, "strides", model_strides)
        strides = model_strides or data_cfg.get("strides", [8, 16, 32])

        self.assigner = TargetAssigner(
            input_size=data_cfg.get("input_size", 416),
            strides=strides,
            num_classes=cfg["model"]["num_classes"],
        )

        # ── DataLoaders ────────────────────────────────────────────
        self.train_loader, self.val_loader = self._build_dataloaders()

        # ── Checkpoint & Logger ────────────────────────────────────
        self.checkpoint_manager = CheckpointManager(
            save_dir=train_cfg["save_dir"],
            save_interval=train_cfg.get("save_interval", 10),
        )

        wandb_cfg = cfg.get("wandb", {})
        self.logger = MetricLogger(
            project=wandb_cfg.get("project", "vanilla-drone-detector"),
            run_name=wandb_cfg.get("run_name", cfg["model"].get("name", "run")),
            cfg=cfg,
            enabled=wandb_cfg.get("enabled", True),
        )

        self.epochs = train_cfg["epochs"]
        self.grad_clip = train_cfg.get("grad_clip_norm", 10.0)
        self.amp_enabled = train_cfg.get("amp", True) and self.device.type == "cuda"

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, start_epoch: int = 0) -> Any:
        """
        Execute the complete training loop.

        Args:
            start_epoch: Epoch to resume from (0 = train from scratch).
                         When resuming, epochs 1..start_epoch are skipped.
        """
        model_name = self.cfg["model"].get("name", "model")
        print(f"\n{'='*60}")
        print(f"  Training: {model_name}")
        print(f"  Device  : {self.device}")
        print(f"  Epochs  : {self.epochs}")
        if self.use_ema:
            print("  Optimizer: AdamW + Model EMA")
        if start_epoch > 0:
            print(f"  Resuming: from epoch {start_epoch + 1}")
        print(f"{'='*60}\n")

        history = []
        nan_streak = 0
        _MAX_NAN_STREAK = 5  # Abort training after this many consecutive NaN epochs
        train_start_time = time.perf_counter()

        for epoch in range(start_epoch + 1, self.epochs + 1):
            t0_epoch = time.perf_counter()

            # Warmup: linearly ramp LR for the first N epochs
            if epoch <= self.warmup_epochs:
                self._apply_warmup_lr(epoch)

            train_metrics = self.train_one_epoch(epoch)

            # ── NaN / Inf loss guard ──────────────────────────────────
            if not math.isfinite(train_metrics.get("total_loss", 0.0)):
                nan_streak += 1
                warnings.warn(
                    f"[Trainer] Epoch {epoch}: NaN/Inf training loss detected "
                    f"(streak {nan_streak}/{_MAX_NAN_STREAK}). Skipping checkpoint.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                if nan_streak >= _MAX_NAN_STREAK:
                    print(f"\n[!] ABORTING: {_MAX_NAN_STREAK} consecutive NaN epochs. "
                          f"Check learning rate, data, or loss configuration.")
                    break
                continue
            nan_streak = 0  # Reset on healthy epoch

            # ── Validation (protected against unexpected crashes) ─────
            try:
                val_metrics = self.validate()
            except Exception as e:
                warnings.warn(
                    f"[Trainer] Epoch {epoch}: Validation failed with {type(e).__name__}: {e}. "
                    f"Using placeholder metrics.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                val_metrics = {"mAP50": 0.0, "mAP50_95": 0.0, "precision": 0.0, "recall": 0.0, "latency_ms": 0.0, "fps": 0.0}

            if epoch > self.warmup_epochs:
                # On the first post-warmup epoch, reset scheduler's internal
                # last_epoch so the cosine cycle starts fresh from base_lr.
                if epoch == self.warmup_epochs + 1:
                    self.scheduler.last_epoch = -1
                self.scheduler.step()

            epoch_sec = time.perf_counter() - t0_epoch
            all_metrics = {**train_metrics, **val_metrics, "epoch_time_sec": epoch_sec}
            history.append({"epoch": epoch, **all_metrics})

            # Save history, checkpoints, and logs on rank 0 only
            if self.rank == 0:
                try:
                    save_dir = Path(self.cfg["training"]["save_dir"])
                    save_dir.mkdir(parents=True, exist_ok=True)
                    with open(save_dir / "history.json", "w") as f:
                        json.dump(history, f, indent=2)
                    
                    if history:
                        with open(save_dir / "history.csv", "w", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
                            writer.writeheader()
                            writer.writerows(history)
                except OSError as e:
                    warnings.warn(
                        f"[Trainer] Failed to write history to disk: {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

                # Save checkpoint (using EMA weights if enabled for higher evaluation quality)
                save_model = self.ema.ema if self.ema is not None else self.model
                self.checkpoint_manager.save(
                    epoch, save_model, self.optimizer, self.scheduler, val_metrics
                )

                self.logger.log(
                    {f"train/{k}": v for k, v in train_metrics.items()}
                    | {f"val/{k}": v for k, v in val_metrics.items()}
                    | {"lr": self.optimizer.param_groups[0]["lr"]},
                    step=epoch,
                )

                self._print_epoch_summary(epoch, all_metrics)

        total_train_sec = time.perf_counter() - train_start_time
        mins, secs = divmod(int(total_train_sec), 60)
        avg_epoch_sec = total_train_sec / max(self.epochs - start_epoch, 1)

        if self.rank == 0:
            try:
                save_dir = Path(self.cfg["training"]["save_dir"])
                summary = {
                    "model_name": model_name,
                    "epochs": self.epochs,
                    "total_train_time_sec": total_train_sec,
                    "total_train_time_formatted": f"{mins}m {secs}s",
                    "avg_epoch_sec": avg_epoch_sec,
                    "best_epoch": getattr(self.checkpoint_manager, "best_epoch", 0),
                    "best_mAP50": getattr(self.checkpoint_manager, "best_score", getattr(self.checkpoint_manager, "best_metric", 0.0)),
                }
                with open(save_dir / "train_summary.json", "w") as f:
                    json.dump(summary, f, indent=2)
            except Exception as e:
                warnings.warn(f"[Trainer] Could not write train_summary.json: {e}")

            self.logger.finish()
            print(f"\n[✓] Training complete in {mins}m {secs}s ({avg_epoch_sec:.2f}s/epoch).")

        if self.is_distributed:
            dist.destroy_process_group()

        return history

    # ──────────────────────────────────────────────────────────────────────────
    # Training / Validation steps
    # ──────────────────────────────────────────────────────────────────────────

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one full training epoch."""
        self.model.train()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        total_loss = cls_loss = reg_loss = obj_loss = 0.0
        num_batches = len(self.train_loader)
        if num_batches == 0:
            raise RuntimeError(
                f"DataLoader has 0 batches! Dataset size is {len(self.train_loader.dataset)}, "
                f"batch size is {self.cfg['training']['batch_size']}. Please ensure data was split properly."
            )

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch:03d}/{self.epochs}",
            leave=False,
            disable=(self.rank != 0),
        )

        for images, batch_targets, _, _ in pbar:
            images = images.to(self.device, non_blocking=True).float() / 255.0
            batch_targets = self._targets_to_device(batch_targets)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                outputs = self.model(images)
                loss, loss_dict = self.criterion(outputs, batch_targets)

            self.scaler.scale(loss).backward()

            # Gradient clipping prevents exploding gradients from random init
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.ema is not None:
                self.ema.update(self.model)

            total_loss += loss.item()
            cls_loss += loss_dict["loss_cls"]
            reg_loss += loss_dict["loss_reg"]
            obj_loss += loss_dict["loss_obj"]

            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "reg": f"{loss_dict['loss_reg']:.3f}",
            })

        return {
            "total_loss": total_loss / num_batches,
            "loss_cls": cls_loss / num_batches,
            "loss_reg": reg_loss / num_batches,
            "loss_obj": obj_loss / num_batches,
        }

    def validate(self) -> Dict[str, float]:
        """Run evaluation on the validation set."""
        eval_model = self.ema.ema if self.ema is not None else self.model
        evaluator = Evaluator(
            model=eval_model,
            val_loader=self.val_loader,
            num_classes=self.cfg["model"]["num_classes"],
            device=self.device,
            assigner=self.assigner,
        )
        return evaluator.evaluate()

    # ──────────────────────────────────────────────────────────────────────────
    # Builder helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_device(self) -> torch.device:
        """Select CUDA (with local_rank if DDP) or CPU."""
        if self.is_distributed and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            device = torch.device(f"cuda:{self.local_rank}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.rank == 0:
            print(f"[*] Using device: {device}")
            if device.type == "cuda":
                print(f"    GPU count: {torch.cuda.device_count()}")
                print(f"    GPU name : {torch.cuda.get_device_name(0)}")
        return device

    def _build_model(self) -> nn.Module:
        """Instantiate detector model with DDP or DataParallel."""
        model = VanillaDroneDetector(self.cfg["model"]).to(self.device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        if self.is_distributed:
            if self.rank == 0:
                print(f"[*] Multi-GPU: Initializing DDP across {self.world_size} processes (NCCL backend)")
            model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)
        elif self.device.type == "cuda" and torch.cuda.device_count() > 1:
            print(f"[*] Multi-GPU: Active on {torch.cuda.device_count()} GPUs via DataParallel")
            model = nn.DataParallel(model)
            
        if self.rank == 0:
            print(f"[*] Model: {self.cfg['model'].get('name', 'VanillaDroneDetector')} "
                  f"| Params: {num_params:,}")
        return model

    def _build_scheduler(self, epochs: int, sched_cfg: dict) -> Any:
        """Build a LR scheduler from config."""
        sched_type = sched_cfg.get("type", "cosine")
        # T_max should only count the non-warmup epochs so the cosine
        # cycle spans exactly (epochs - warmup_epochs) steps.
        cosine_epochs = max(epochs - self.warmup_epochs, 1)
        if sched_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cosine_epochs,
                eta_min=sched_cfg.get("eta_min", 1e-6),
            )
        elif sched_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_cfg.get("step_size", 30),
                gamma=sched_cfg.get("gamma", 0.1),
            )
        raise ValueError(f"Unknown scheduler type: {sched_type}")

    def _build_dataloaders(self):
        """Build train and val DataLoaders."""
        data_cfg = self.cfg["data"]
        aug_cfg = self.cfg.get("augmentation", {})

        train_transforms = build_transforms(aug_cfg, is_train=True)
        val_transforms = build_transforms(aug_cfg, is_train=False)

        train_dataset = DroneYOLODataset(
            manifest_path=data_cfg["train_manifest"],
            input_size=data_cfg.get("input_size", 416),
            transforms=train_transforms,
        )
        val_dataset = DroneYOLODataset(
            manifest_path=data_cfg["val_manifest"],
            input_size=data_cfg.get("input_size", 416),
            transforms=val_transforms,
        )

        if len(train_dataset) == 0:
            raise RuntimeError(f"Train manifest is empty: {data_cfg['train_manifest']}. Please check Section 4 dataset split.")
        if len(val_dataset) == 0:
            raise RuntimeError(f"Val manifest is empty: {data_cfg['val_manifest']}. Please check Section 4 dataset split.")

        if self.rank == 0:
            print(f"[*] Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

        num_workers = data_cfg.get("num_workers", 2)
        batch_size = self.cfg["training"]["batch_size"]
        use_cuda = self.device.type == "cuda"

        # Drop last if dataset has more than 1 batch to avoid 0-batch training
        drop_last = len(train_dataset) > batch_size

        nw = num_workers if use_cuda else 0
        kwargs = {}
        if nw > 0:
            kwargs["persistent_workers"] = data_cfg.get("persistent_workers", False)
            kwargs["prefetch_factor"] = data_cfg.get("prefetch_factor", 2)

        # In DDP, use DistributedSampler to partition data across ranks
        self.train_sampler = None
        if self.is_distributed:
            self.train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            shuffle = False
        else:
            shuffle = True

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=self.train_sampler,
            num_workers=nw,
            collate_fn=self.assigner.collate_and_assign,
            pin_memory=use_cuda,
            drop_last=drop_last,
            **kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=nw,
            collate_fn=self.assigner.collate_and_assign,
            pin_memory=use_cuda,
            **kwargs,
        )

        return train_loader, val_loader

    def _targets_to_device(
        self,
        batch_targets,
    ):
        """Move all target tensors in the batch to the training device."""
        return [
            {k: v.to(self.device, non_blocking=True) for k, v in level.items()}
            for level in batch_targets
        ]

    def _apply_warmup_lr(self, epoch: int) -> None:
        """Linear LR warmup for the first `warmup_epochs` epochs."""
        base_lr = self.cfg["training"]["lr"]
        warmup_lr = base_lr * (epoch / self.warmup_epochs)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = warmup_lr

    @staticmethod
    def _print_epoch_summary(epoch: int, metrics: Dict[str, float]) -> None:
        """Print a compact one-line epoch summary."""
        parts = [f"Epoch {epoch:03d}"]
        for k, v in metrics.items():
            parts.append(f"{k}={v:.4f}")
        print("  " + " | ".join(parts))
