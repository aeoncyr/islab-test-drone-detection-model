"""
benchmark.py — Master Benchmark & Cross-Model Comparison CLI.

Evaluates all trained models (Model-A, Model-B, Model-C, YOLOv8, RT-DETR),
measures accuracy metrics (mAP@50, mAP@50:95, Precision, Recall), parameter count,
and real-time inference latency (FPS / ms), then exports publication-ready tables & charts.

Usage:
    # Benchmark all available checkpoints in runs/
    python benchmark.py

    # Benchmark specific checkpoints and custom manifest
    python benchmark.py --val_manifest splits/val.txt --output_dir runs/benchmark_results
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import DroneYOLODataset
from src.engine.evaluator import Evaluator
from src.engine.target_assigner import TargetAssigner
from src.models.detector import VanillaDroneDetector
from src.utils.checkpoint import CheckpointManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Master benchmark and comparison suite for drone detection models."
    )
    parser.add_argument(
        "--val_manifest",
        type=str,
        default="splits/val.txt",
        help="Path to validation split manifest. Default: splits/val.txt",
    )
    parser.add_argument(
        "--runs_dir",
        type=str,
        default="runs",
        help="Directory containing trained model runs. Default: runs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/benchmark_results",
        help="Directory to save comparison tables and charts. Default: runs/benchmark_results",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=416,
        help="Inference image resolution. Default: 416",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Validation batch size. Default: 16",
    )
    parser.add_argument(
        "--benchmark_latency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Measure inference latency and FPS for each model (use --no-benchmark_latency to disable).",
    )
    return parser.parse_args()


def measure_latency_and_fps(
    model: torch.nn.Module,
    device: torch.device,
    input_size: int = 416,
    num_warmup: int = 20,
    num_runs: int = 100,
) -> Dict[str, float]:
    """Measure inference latency (ms) and throughput (FPS)."""
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)
    model.eval()

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_runs):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
        total_time = time.perf_counter() - start

    avg_ms = (total_time / num_runs) * 1000.0
    fps = num_runs / total_time
    return {"latency_ms": avg_ms, "fps": fps}


def evaluate_custom_model(
    label: str,
    cfg_m: Dict[str, Any],
    ckpt_path: Path,
    val_loader: DataLoader,
    device: torch.device,
    measure_fps: bool = True,
) -> Optional[Dict[str, Any]]:
    """Evaluate a custom VanillaDroneDetector checkpoint."""
    if not ckpt_path.exists():
        return None

    print(f"[*] Evaluating {label} from {ckpt_path} ...")
    model = VanillaDroneDetector(cfg_m).to(device)
    CheckpointManager.load(str(ckpt_path), model, device=device)

    evaluator = Evaluator(model, val_loader, num_classes=1, device=device)
    metrics = evaluator.evaluate()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    perf = measure_latency_and_fps(model, device) if measure_fps else {"latency_ms": metrics.get("latency_ms", 0.0), "fps": metrics.get("fps", 0.0)}

    # Attempt to load training duration from train_summary.json
    train_time_str = "N/A"
    train_summary_path = ckpt_path.parent / "train_summary.json"
    if train_summary_path.exists():
        try:
            with open(train_summary_path) as f:
                s = json.load(f)
                train_time_str = s.get("total_train_time_formatted", "N/A")
        except Exception:
            pass

    return {
        "params_m": params,
        "mAP50": metrics.get("mAP50", 0.0),
        "mAP50_95": metrics.get("mAP50_95", 0.0),
        "precision": metrics.get("precision", 0.0),
        "recall": metrics.get("recall", 0.0),
        "latency_ms": perf["latency_ms"],
        "fps": perf["fps"],
        "train_time": train_time_str,
    }


def evaluate_ultralytics_baseline(
    label: str,
    weights_path: Path,
    data_yaml: Path,
    imgsz: int = 416,
) -> Optional[Dict[str, Any]]:
    """Evaluate a pretrained Ultralytics baseline (YOLO / RT-DETR)."""
    if not weights_path.exists():
        return None

    try:
        from ultralytics import YOLO, RTDETR
    except ImportError:
        print("[!] Ultralytics not installed. Skipping baseline evaluation.")
        return None

    print(f"[*] Evaluating baseline {label} from {weights_path} ...")
    if "rtdetr" in str(weights_path).lower():
        model = RTDETR(str(weights_path))
    else:
        model = YOLO(str(weights_path))

    val_res = model.val(data=str(data_yaml), imgsz=imgsz, verbose=False)
    
    # Extract params
    params = sum(p.numel() for p in model.model.parameters()) / 1e6

    # Latency / FPS from Ultralytics speed dict if available
    latency_ms = val_res.speed.get("inference", 0.0)
    fps = (1000.0 / latency_ms) if latency_ms > 0 else 0.0

    return {
        "params_m": params,
        "mAP50": float(val_res.box.map50),
        "mAP50_95": float(val_res.box.map),
        "precision": float(val_res.box.mp),
        "recall": float(val_res.box.mr),
        "latency_ms": latency_ms,
        "fps": fps,
        "train_time": "N/A",
    }


def df_to_markdown_fallback(df: pd.DataFrame) -> str:
    """Safely convert DataFrame to Markdown table without requiring tabulate."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        headers = [str(col) for col in df.columns]
        col_widths = [max(len(str(val)) for val in [h] + list(df[col])) for h, col in zip(headers, df.columns)]
        header_row = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
        sep_row = "| " + " | ".join("-" * w for w in col_widths) + " |"
        data_rows = [
            "| " + " | ".join(str(val).ljust(w) for val, w in zip(row, col_widths)) + " |"
            for row in df.itertuples(index=False)
        ]
        return "\n".join([header_row, sep_row] + data_rows)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("      DRONE DETECTION MASTER BENCHMARK & COMPARISON SUITE")
    print(f"{'='*70}")
    print(f"[*] Device         : {device}")
    print(f"[*] Validation Split: {args.val_manifest}")
    print(f"[*] Runs Directory : {runs_dir}")
    print(f"[*] Output Directory: {out_dir}")

    # Build validation DataLoader for custom models
    assigner = TargetAssigner(input_size=args.input_size, strides=[4, 8, 16, 32], num_classes=1)
    val_dataset = DroneYOLODataset(manifest_path=args.val_manifest, input_size=args.input_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=assigner.collate_and_assign,
        pin_memory=(device.type == "cuda"),
    )

    models_info = [
        {
            "label": "Model-A",
            "paradigm": "Vanilla CNN",
            "backbone": "CSPDarknet",
            "neck": "FPN",
            "attention": "None",
            "weights": "Scratch (Vanilla)",
            "cfg": {"num_classes": 1, "base_channels": 32, "neck_type": "fpn", "use_cbam": False},
            "ckpt": runs_dir / "model_a_fpn" / "best.pth",
            "type": "custom",
        },
        {
            "label": "Model-B",
            "paradigm": "Vanilla CNN",
            "backbone": "CSPDarknet",
            "neck": "FPN + PAN",
            "attention": "None",
            "weights": "Scratch (Vanilla)",
            "cfg": {"num_classes": 1, "base_channels": 32, "neck_type": "fpnpan", "use_cbam": False},
            "ckpt": runs_dir / "model_b_pan" / "best.pth",
            "type": "custom",
        },
        {
            "label": "Model-C",
            "paradigm": "Vanilla CNN",
            "backbone": "CSPDarknet",
            "neck": "FPN + PAN",
            "attention": "CBAM",
            "weights": "Scratch (Vanilla)",
            "cfg": {"num_classes": 1, "base_channels": 32, "neck_type": "fpnpan", "use_cbam": True},
            "ckpt": runs_dir / "model_c_attn" / "best.pth",
            "type": "custom",
        },
        {
            "label": "Model-D",
            "paradigm": "Vanilla CNN",
            "backbone": "CSPDarknet",
            "neck": "FPN + PAN (4-Level)",
            "attention": "CBAM + EMA",
            "weights": "Scratch (Vanilla)",
            "cfg": {"num_classes": 1, "base_channels": 32, "neck_type": "fpnpan4", "use_cbam": True},
            "ckpt": runs_dir / "model_d_p2_ema" / "best.pth",
            "type": "custom",
        },
        {
            "label": "YOLOv8-nano",
            "paradigm": "Pretrained CNN",
            "backbone": "Modified CSPNet",
            "neck": "PAN",
            "attention": "None",
            "weights": "Pretrained (COCO)",
            "ckpt": runs_dir / "baselines" / "yolov8n_baseline" / "weights" / "best.pt",
            "type": "ultralytics",
        },
        {
            "label": "YOLOv8-small",
            "paradigm": "Pretrained CNN",
            "backbone": "Modified CSPNet",
            "neck": "PAN",
            "attention": "None",
            "weights": "Pretrained (COCO)",
            "ckpt": runs_dir / "baselines" / "yolov8s_baseline" / "weights" / "best.pt",
            "type": "ultralytics",
        },
        {
            "label": "RT-DETR-L",
            "paradigm": "Vision Transformer",
            "backbone": "HGNetv2",
            "neck": "Hybrid Encoder",
            "attention": "Transformer Cross-Attn",
            "weights": "Pretrained (COCO)",
            "ckpt": runs_dir / "baselines" / "rtdetr_l_baseline" / "weights" / "best.pt",
            "type": "ultralytics",
        },
    ]

    # Locate or auto-generate data YAML for Ultralytics baselines
    data_yaml = Path("drone_yolo.yaml")
    if not data_yaml.exists():
        data_yaml = Path("configs/drone_yolo.yaml")
    if not data_yaml.exists():
        data_yaml = Path("baselines/drone_yolov8.yaml")

    rows = []
    for info in models_info:
        if info["type"] == "custom":
            res = evaluate_custom_model(
                label=info["label"],
                cfg_dict=info["cfg"],
                ckpt_path=info["ckpt"],
                val_loader=val_loader,
                device=device,
                input_size=args.input_size,
                benchmark_latency=args.benchmark_latency,
            )
        else:
            res = evaluate_ultralytics_baseline(
                label=info["label"],
                weights_path=info["ckpt"],
                data_yaml=data_yaml,
                imgsz=args.input_size,
            )

        if res is not None:
            rows.append({
                "Model": info["label"],
                "Paradigm": info["paradigm"],
                "Backbone": info["backbone"],
                "Neck": info["neck"],
                "Attention": info["attention"],
                "Params (M)": f"{res['params_m']:.2f}M",
                "mAP@50 (%)": f"{res['mAP50'] * 100:.2f}%" if res['mAP50'] <= 1.0 else f"{res['mAP50']:.2f}%",
                "mAP@50:95 (%)": f"{res['mAP50_95'] * 100:.2f}%" if res['mAP50_95'] <= 1.0 else f"{res['mAP50_95']:.2f}%",
                "Precision (%)": f"{res['precision'] * 100:.2f}%" if res['precision'] <= 1.0 else f"{res['precision']:.2f}%",
                "Recall (%)": f"{res['recall'] * 100:.2f}%" if res['recall'] <= 1.0 else f"{res['recall']:.2f}%",
                "Train Time": res.get("train_time", "N/A"),
                "Latency (ms)": f"{res['latency_ms']:.2f} ms" if res['latency_ms'] > 0 else "N/A",
                "FPS": f"{res['fps']:.1f}" if res['fps'] > 0 else "N/A",
                "Training Type": info["weights"],
            })

    df = pd.DataFrame(rows)

    print("\n" + "=" * 110)
    print("                           MASTER MODEL BENCHMARK & COMPARISON TABLE")
    print("=" * 110)
    if not df.empty:
        print(df.to_string(index=False))
    else:
        print("[!] No evaluated models found. Please train models or provide valid checkpoint paths.")
    print("=" * 110)

    # Save to CSV and Markdown
    csv_path = out_dir / "model_comparison_table.csv"
    md_path = out_dir / "model_comparison_table.md"
    if not df.empty:
        df.to_csv(csv_path, index=False)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(df_to_markdown_fallback(df))
        print(f"[✓] Master comparison table saved to:\n    - {csv_path}\n    - {md_path}")

    # Generate Publication 2-Panel Figure: Bar Chart + Speed-Accuracy Pareto Frontier
    if len(rows) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 6.5))

        # Panel 1: Bar Chart
        models = [r["Model"] for r in rows]
        map50_vals = [float(r["mAP@50 (%)"].replace("%", "")) for r in rows]
        map50_95_vals = [float(r["mAP@50:95 (%)"].replace("%", "")) for r in rows]

        x = np.arange(len(models))
        width = 0.35

        rects1 = ax1.bar(x - width / 2, map50_vals, width, label="mAP@50 (%)", color="#2b5c8f", edgecolor="black", alpha=0.9)
        rects2 = ax1.bar(x + width / 2, map50_95_vals, width, label="mAP@50:95 (%)", color="#e05a47", edgecolor="black", alpha=0.9)

        ax1.set_ylabel("mAP Score (%)", fontsize=12, fontweight="bold")
        ax1.set_title("(a) Detection Accuracy Across Architectures", fontsize=13, fontweight="bold", pad=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, fontsize=9.0, fontweight="bold")
        ax1.legend(fontsize=10.5, loc="upper left")
        ax1.grid(axis="y", linestyle="--", alpha=0.6)
        ax1.set_ylim(0, 105)

        for rect in rects1:
            h = rect.get_height()
            ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        for rect in rects2:
            h = rect.get_height()
            ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        # Panel 2: Speed-Accuracy Pareto Frontier
        fps_vals, map_strict, labels, params, colors_scatter = [], [], [], [], []
        color_map = {
            "Model-A": "#1f77b4", "Model-B": "#ff7f0e", "Model-C": "#2ca02c",
            "Model-D": "#9467bd", "YOLOv8-nano": "#d62728", "YOLOv8-small": "#8c564b", "RT-DETR-L": "#e377c2"
        }
        for r in rows:
            fps_str = r.get("FPS", "N/A")
            if fps_str != "N/A":
                fps_vals.append(float(fps_str))
                map_strict.append(float(r["mAP@50:95 (%)"].replace("%", "")))
                p_val = float(r["Params (M)"].replace("M", ""))
                m_label = r["Model"]
                labels.append(m_label)
                params.append(p_val)
                colors_scatter.append(color_map.get(m_label, "#333333"))

        if fps_vals:
            bubble_sizes = [max(p * 25, 80) for p in params]
            ax2.scatter(fps_vals, map_strict, s=bubble_sizes, c=colors_scatter, alpha=0.75, edgecolors="black", linewidth=1.5)
            label_offsets = {
                "Model-A": (15, -2.5), "Model-B": (-120, -2.5), "Model-C": (15, 2.2),
                "Model-D": (18, -1.2), "YOLOv8-nano": (-125, 1.8), "YOLOv8-small": (-45, -3.2), "RT-DETR-L": (18, -0.5),
            }
            for i, txt in enumerate(labels):
                dx, dy = label_offsets.get(txt, (12, 1.5))
                ax2.annotate(
                    f"{txt} ({params[i]:.1f}M)",
                    (fps_vals[i], map_strict[i]),
                    xytext=(fps_vals[i] + dx, map_strict[i] + dy),
                    fontsize=8.5,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bbbbbb", alpha=0.85),
                    arrowprops=dict(arrowstyle="->", color="#444444", lw=0.9),
                )
            ax2.set_xlabel("Inference Speed (FPS) [Higher is Faster →]", fontsize=12, fontweight="bold")
            ax2.set_ylabel("Strict COCO mAP@50:95 (%) [Higher is Better ↑]", fontsize=12, fontweight="bold")
            ax2.set_title("(b) Efficiency vs. Precision Pareto Frontier", fontsize=13, fontweight="bold", pad=12)
            ax2.grid(True, linestyle="--", alpha=0.6)
            ax2.set_ylim(min(map_strict) - 6, max(map_strict) + 5)
            ax2.axvline(x=30, color="#888888", linestyle=":", linewidth=1.5)
            ax2.text(33, min(map_strict) - 4.5, "Real-Time Threshold (30 FPS)", fontsize=8.0, color="#666666", style="italic")

        plt.tight_layout()
        chart_path = out_dir / "model_comparison_chart.png"
        plt.savefig(chart_path, dpi=300)
        plt.show()
        print(f"[✓] 2-Panel benchmark comparison figure saved to: {chart_path}")


if __name__ == "__main__":
    main()
