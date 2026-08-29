"""
infer.py — Single-image inference with visualization.

Runs the detector on one image and saves/shows the result with
bounding boxes drawn on top.

Usage:
    python infer.py \\
        --config configs/model_c_attn.yaml \\
        --checkpoint runs/model_c_attn/best.pth \\
        --image datasets/sample/augmented_raw_dataset_city_foggy_city_foggy_0_1_sequence.4_step0.camera.png \\
        --output output_detection.jpg

    # Adjust confidence threshold
    python infer.py --config ... --checkpoint ... --image ... --conf 0.3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from src.models.detector import VanillaDroneDetector
from src.utils.box_ops import decode_predictions, batched_nms
from src.utils.checkpoint import CheckpointManager


CLASS_NAMES = ["drone"]
# Distinct color per class (BGR-style; PIL uses RGB)
CLASS_COLORS = ["#FF4B4B"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-image inference and visualize detections."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument(
        "--output",
        type=str,
        default="detection_result.jpg",
        help="Path to save the annotated image.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS IoU threshold.")
    return parser.parse_args()


def preprocess_image(
    image_path: str,
    input_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load and resize image → (1, 3, H, W) tensor on device."""
    img = Image.open(image_path).convert("RGB")
    orig_size = img.size  # (W, H)
    img_resized = img.resize((input_size, input_size), Image.BILINEAR)
    tensor = torch.from_numpy(
        np.array(img_resized, dtype=np.float32)
    ).permute(2, 0, 1).unsqueeze(0) / 255.0
    return tensor.to(device), orig_size


def draw_detections(
    image_path: str,
    detections: torch.Tensor,
    orig_size: tuple[int, int],
    input_size: int,
    conf_threshold: float,
) -> Image.Image:
    """
    Draw detected boxes on the original-resolution image.

    Args:
        detections: (K, 6) tensor [x1, y1, x2, y2, conf, class_id] at input_size scale.
        orig_size : (W, H) of the original image.
        input_size: Resolution the model ran at (for coordinate scaling).

    Returns:
        PIL Image with bounding boxes drawn.
    """
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig_size
    scale_x = orig_w / input_size
    scale_y = orig_h / input_size

    draw = ImageDraw.Draw(img)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", size=max(12, orig_h // 60))
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det.tolist()
        if conf < conf_threshold:
            continue

        # Scale back to original resolution
        x1, x2 = x1 * scale_x, x2 * scale_x
        y1, y2 = y1 * scale_y, y2 * scale_y

        cls_id = int(cls_id)
        label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        label_text = f"{label} {conf:.2f}"

        # Draw bounding box
        line_width = max(2, orig_h // 300)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Draw label background
        text_bbox = draw.textbbox((x1, y1), label_text, font=font)
        draw.rectangle(
            [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
            fill=color,
        )
        draw.text((x1, y1), label_text, fill="white", font=font)

    return img


def main() -> None:
    args = parse_args()

    # Validate input files
    for name, path in [("config", args.config), ("checkpoint", args.checkpoint), ("image", args.image)]:
        if not Path(path).exists():
            print(f"[ERROR] {name} file not found: {path}")
            sys.exit(1)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = cfg["data"].get("input_size", 416)

    # Build and load model
    try:
        model = VanillaDroneDetector(cfg["model"]).to(device)
        CheckpointManager.load(args.checkpoint, model, device=device)
        model.eval()
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    # Preprocess
    image_tensor, orig_size = preprocess_image(args.image, input_size, device)

    # Inference with torch.inference_mode for better performance than no_grad
    with torch.inference_mode():
        outputs = model(image_tensor)

    # Post-process
    decoded = decode_predictions(outputs, conf_threshold=args.conf)
    detections = decoded[0]  # Single image
    detections = batched_nms(detections.cpu(), iou_threshold=args.nms)

    print(f"[*] Detected {len(detections)} object(s) with conf ≥ {args.conf}")
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        label = CLASS_NAMES[int(cls)] if int(cls) < len(CLASS_NAMES) else str(int(cls))
        print(f"    {label}: conf={conf:.3f}  box=[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Draw and save
    result_img = draw_detections(
        args.image, detections, orig_size, input_size, args.conf
    )
    result_img.save(str(output_path))
    print(f"[✓] Saved annotated image → {output_path}")


if __name__ == "__main__":
    main()
