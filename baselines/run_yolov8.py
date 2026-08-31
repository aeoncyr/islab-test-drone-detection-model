"""
baselines/run_yolov8.py

YOLOv8-nano baseline: fine-tune on the drone dataset and evaluate.

This script is used ONLY for generating comparison results in the paper.
The VanillaDroneDetector is our submission model (trained from scratch).
YOLOv8 uses pretrained COCO weights, giving it a significant head start —
making the comparison favorable to our from-scratch approach.

Requirements:
    pip install ultralytics

Usage:
    # Fine-tune YOLOv8-nano on drone dataset
    python baselines/run_yolov8.py \\
        --data_yaml baselines/drone_yolov8.yaml \\
        --epochs 50 \\
        --imgsz 416 \\
        --batch 8 \\
        --run_name yolov8n_drone_baseline
"""

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8-nano on drone dataset for paper comparison."
    )
    parser.add_argument(
        "--data_yaml",
        type=str,
        default="baselines/drone_yolov8.yaml",
        help="Path to YOLOv8-format dataset YAML.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--run_name", type=str, default="yolov8n_baseline")
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Pretrained YOLOv8 weights (downloaded automatically by Ultralytics).",
    )
    return parser.parse_args()


def generate_data_yaml(output_path: str, train_manifest: str, val_manifest: str) -> None:
    """Generate a YOLOv8-compatible dataset YAML from our split manifests."""
    import yaml

    data = {
        "train": str(Path(train_manifest).resolve()),
        "val": str(Path(val_manifest).resolve()),
        "nc": 1,
        "names": {0: "drone"},
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"[*] YOLOv8 data YAML written → {output_path}")


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        return

    # Auto-generate data YAML if it does not exist
    if not Path(args.data_yaml).exists():
        print("[*] Generating YOLOv8 data YAML from split manifests...")
        generate_data_yaml(args.data_yaml, "splits/train.txt", "splits/val.txt")

    print(f"\n[*] Fine-tuning YOLOv8-nano ({args.weights}) on drone dataset")
    print(f"    Epochs: {args.epochs} | ImgSz: {args.imgsz} | Batch: {args.batch}")

    model = YOLO(args.weights)

    results = model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.run_name,
        project="runs/baselines",
        exist_ok=True,
        verbose=True,
    )

    print("\n[*] Running validation...")
    val_metrics = model.val()

    print("\n" + "=" * 50)
    print("  YOLOv8-nano Baseline Results")
    print("=" * 50)
    print(f"  mAP@50    : {val_metrics.box.map50:.4f}")
    print(f"  mAP@50:95 : {val_metrics.box.map:.4f}")
    print(f"  Precision : {val_metrics.box.mp:.4f}")
    print(f"  Recall    : {val_metrics.box.mr:.4f}")
    print("=" * 50)
    print(f"\n  Best weights: runs/baselines/{args.run_name}/weights/best.pt")


if __name__ == "__main__":
    main()
