"""
baselines/run_rtdetr.py

RT-DETR-R18 baseline: fine-tune on the drone dataset and evaluate.

RT-DETR (Real-Time DEtection TRansformer) is a transformer-based detector
that serves as a complementary comparison to the CNN-based YOLOv8.
This provides a richer ablation for the paper: CNN pretrained, Transformer
pretrained, and our vanilla CNN from-scratch.

Requirements:
    pip install ultralytics

Usage:
    python baselines/run_rtdetr.py \\
        --data_yaml baselines/drone_yolov8.yaml \\
        --epochs 50 \\
        --imgsz 416 \\
        --batch 8 \\
        --run_name rtdetr_r18_baseline
"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune RT-DETR-R18 on drone dataset for paper comparison."
    )
    parser.add_argument("--data_yaml", type=str, default="baselines/drone_yolov8.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=4, help="Transformer needs smaller batch.")
    parser.add_argument("--run_name", type=str, default="rtdetr_r18_baseline")
    parser.add_argument("--weights", type=str, default="rtdetr-r18.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import RTDETR
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        return

    # Check if data YAML exists (run run_yolov8.py first if not)
    if not Path(args.data_yaml).exists():
        print(
            f"[!] Data YAML not found: {args.data_yaml}\n"
            f"    Run: python baselines/run_yolov8.py --data_yaml {args.data_yaml} first."
        )
        return

    print(f"\n[*] Fine-tuning RT-DETR-R18 ({args.weights}) on drone dataset")
    print(f"    Epochs: {args.epochs} | ImgSz: {args.imgsz} | Batch: {args.batch}")

    model = RTDETR(args.weights)

    model.train(
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
    print("  RT-DETR-R18 Baseline Results")
    print("=" * 50)
    print(f"  mAP@50    : {val_metrics.box.map50:.4f}")
    print(f"  mAP@50:95 : {val_metrics.box.map:.4f}")
    print(f"  Precision : {val_metrics.box.mp:.4f}")
    print(f"  Recall    : {val_metrics.box.mr:.4f}")
    print("=" * 50)
    print(f"\n  Best weights: runs/baselines/{args.run_name}/weights/best.pt")


if __name__ == "__main__":
    main()
