"""
train.py — Training entry point.

Thin script that loads a YAML config and delegates all logic to the Trainer class.

Usage:
    # Train Model-A (FPN baseline)
    python train.py --config configs/model_a_fpn.yaml

    # Train Model-B (FPN+PAN)
    python train.py --config configs/model_b_pan.yaml

    # Train Model-C (FPN+PAN + CBAM attention) — expected best
    python train.py --config configs/model_c_attn.yaml

    # Override any config value via CLI
    python train.py --config configs/model_a_fpn.yaml training.epochs=2 training.batch_size=2

    # Resume from checkpoint
    python train.py --config configs/model_a_fpn.yaml --resume runs/model_a_fpn/latest.pth
"""

import argparse
import sys
from pathlib import Path

import yaml

from src.engine.trainer import Trainer
from src.utils.checkpoint import CheckpointManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Vanilla Drone Detector from scratch."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g. configs/model_a_fpn.yaml).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pth file to resume training from.",
    )
    # Allow CLI overrides: e.g. training.epochs=5
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional config overrides in key=value format (e.g. training.batch_size=4).",
    )
    return parser.parse_args()


def apply_overrides(cfg: dict, overrides: list) -> dict:
    """
    Apply dot-notation CLI overrides to the config dict.

    Example:
        "training.epochs=50"  →  cfg["training"]["epochs"] = 50
    """
    for override in overrides:
        if "=" not in override:
            print(f"  [!] Skipping invalid override (no '='): {override}")
            continue
        key_path, value_str = override.split("=", 1)
        keys = key_path.split(".")

        # Try to cast value to int/float/bool
        try:
            value = int(value_str)
        except ValueError:
            try:
                value = float(value_str)
            except ValueError:
                if value_str.lower() == "true":
                    value = True
                elif value_str.lower() == "false":
                    value = False
                else:
                    value = value_str

        # Navigate into nested dict
        sub = cfg
        for k in keys[:-1]:
            sub = sub.setdefault(k, {})
        sub[keys[-1]] = value
        print(f"  [Override] {key_path} = {value}")

    return cfg


def main() -> None:
    args = parse_args()

    # Load YAML config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Validate required config sections
    _REQUIRED_SECTIONS = ["model", "data", "training"]
    missing = [s for s in _REQUIRED_SECTIONS if s not in cfg]
    if missing:
        print(f"[ERROR] Config missing required sections: {missing}")
        sys.exit(1)

    _REQUIRED_DATA_KEYS = ["train_manifest", "val_manifest"]
    missing_data = [k for k in _REQUIRED_DATA_KEYS if k not in cfg["data"]]
    if missing_data:
        print(f"[ERROR] Config 'data' section missing required keys: {missing_data}")
        sys.exit(1)

    # Apply CLI overrides
    if args.overrides:
        print("[*] Applying CLI overrides:")
        cfg = apply_overrides(cfg, args.overrides)

    print(f"\n[*] Config: {config_path}")
    print(f"    Model  : {cfg['model'].get('name', 'VanillaDroneDetector')}")
    print(f"    Epochs : {cfg['training']['epochs']}")
    print(f"    Batch  : {cfg['training']['batch_size']}")
    print(f"    LR     : {cfg['training']['lr']}")

    try:
        # Build Trainer and run
        trainer = Trainer(cfg)

        start_epoch = 0
        if args.resume:
            print(f"\n[*] Resuming from: {args.resume}")
            state = CheckpointManager.load(
                args.resume,
                trainer.model,
                trainer.optimizer,
                trainer.scheduler,
                trainer.device,
            )
            start_epoch = state.get("epoch", 0)
            print(f"    Resuming from epoch {start_epoch} (next epoch: {start_epoch + 1})")

        trainer.run(start_epoch=start_epoch)

    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user. Saving latest checkpoint...")
        sys.exit(0)
    except Exception as e:
        import traceback
        print(f"\n[FATAL] Training failed with {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()