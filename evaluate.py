"""
evaluate.py — Evaluation entry point.

Loads a trained model checkpoint and computes detection metrics on the
validation split.

Usage:
    python evaluate.py \\
        --config configs/model_c_attn.yaml \\
        --checkpoint runs/model_c_attn/best.pth

    # Evaluate on a different manifest
    python evaluate.py \\
        --config configs/model_a_fpn.yaml \\
        --checkpoint runs/model_a_fpn/best.pth \\
        --manifest splits/val.txt
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

from src.data.dataset import DroneYOLODataset
from src.engine.evaluator import Evaluator
from src.engine.target_assigner import TargetAssigner
from src.models.detector import VanillaDroneDetector
from src.utils.checkpoint import CheckpointManager
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained VanillaDroneDetector checkpoint."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config used during training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the .pth checkpoint file.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Override the val manifest path from config. Default: use config value.",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.05,
        help="Confidence threshold for predictions. Default: 0.05.",
    )
    parser.add_argument(
        "--nms_threshold",
        type=float,
        default=0.45,
        help="IoU threshold for NMS. Default: 0.45.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate inputs
    if not Path(args.config).exists():
        print(f"[ERROR] Config file not found: {args.config}")
        sys.exit(1)
    if not Path(args.checkpoint).exists():
        print(f"[ERROR] Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = cfg["data"]

    # Build model
    try:
        model = VanillaDroneDetector(cfg["model"]).to(device)
        CheckpointManager.load(args.checkpoint, model, device=device)
        model.eval()
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    # Build DataLoader
    manifest = args.manifest or data_cfg["val_manifest"]
    if not Path(manifest).exists():
        print(f"[ERROR] Manifest file not found: {manifest}")
        sys.exit(1)

    strides = getattr(model, "strides", data_cfg.get("strides", [8, 16, 32]))
    assigner = TargetAssigner(
        input_size=data_cfg.get("input_size", 416),
        strides=strides,
        num_classes=cfg["model"]["num_classes"],
    )

    val_dataset = DroneYOLODataset(
        manifest_path=manifest,
        input_size=data_cfg.get("input_size", 416),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=assigner.collate_and_assign,
    )

    print(f"\n[*] Evaluating: {cfg['model'].get('name', 'model')}")
    print(f"    Checkpoint : {args.checkpoint}")
    print(f"    Val images : {len(val_dataset)}")

    evaluator = Evaluator(
        model=model,
        val_loader=val_loader,
        num_classes=cfg["model"]["num_classes"],
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        device=device,
    )

    try:
        metrics = evaluator.evaluate()
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 40)
    print("  Evaluation Results")
    print("=" * 40)
    for k, v in metrics.items():
        print(f"  {k:<15}: {v:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
