"""
src/utils/box_ops.py

Bounding box operations: decoding, NMS, coordinate conversions.
Used by the evaluator and inference pipeline.
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F


def decode_predictions(
    model_outputs: List[dict],
    conf_threshold: float = 0.05,
) -> List[torch.Tensor]:
    """
    Decode raw model outputs into (x1, y1, x2, y2, conf, class_id) per image.

    The objectness and classification scores are combined via:
        confidence = sigmoid(obj) * sigmoid(cls)

    This mirrors FCOS's centerness-weighted classification score.

    Args:
        model_outputs  : List of 3 dicts (P3, P4, P5) from the detector.
        conf_threshold : Minimum confidence to keep a detection. Default: 0.05.

    Returns:
        List of tensors, one per image in the batch.
        Each tensor: (K, 6) with columns [x1, y1, x2, y2, conf, class_id].
        K may be 0 if no detections pass the threshold.
    """
    batch_size = model_outputs[0]["cls"].shape[0]
    all_boxes: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]

    _STRIDES = [4, 8, 16, 32] if len(model_outputs) == 4 else [8, 16, 32]
    for idx, pred in enumerate(model_outputs):
        if "stride" in pred:
            s = pred["stride"]
            stride = int(s.flatten()[0].item()) if isinstance(s, torch.Tensor) else int(s)
        else:
            stride = _STRIDES[idx]
        cls_logits = pred["cls"]   # (B, C, H, W)
        reg_preds = pred["reg"]    # (B, 4, H, W)
        obj_logits = pred["obj"]   # (B, 1, H, W)

        b, c, h, w = cls_logits.shape
        device = cls_logits.device

        # Build grid cell centers
        y_grid, x_grid = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        x_centers = (x_grid.float() + 0.5) * stride  # (H, W)
        y_centers = (y_grid.float() + 0.5) * stride  # (H, W)

        # Decode LTRB → boxes
        l, t, r, bot = reg_preds.split(1, dim=1)
        x1 = x_centers - l.squeeze(1)   # (B, H, W)
        y1 = y_centers - t.squeeze(1)
        x2 = x_centers + r.squeeze(1)
        y2 = y_centers + bot.squeeze(1)

        # Compute confidence score: sigmoid(obj) × sigmoid(cls)
        obj_score = torch.sigmoid(obj_logits)   # (B, 1, H, W)
        cls_score = torch.sigmoid(cls_logits)    # (B, C, H, W)
        conf = (obj_score * cls_score)           # (B, C, H, W)

        for bi in range(b):
            # conf_bi: (C, H, W)
            conf_bi = conf[bi]
            max_conf, class_ids = conf_bi.max(dim=0)  # (H, W) each

            # Flatten
            max_conf_flat = max_conf.reshape(-1)
            class_ids_flat = class_ids.reshape(-1).float()
            x1_flat = x1[bi].reshape(-1)
            y1_flat = y1[bi].reshape(-1)
            x2_flat = x2[bi].reshape(-1)
            y2_flat = y2[bi].reshape(-1)

            keep = max_conf_flat >= conf_threshold
            if keep.sum() == 0:
                continue

            dets = torch.stack([
                x1_flat[keep],
                y1_flat[keep],
                x2_flat[keep],
                y2_flat[keep],
                max_conf_flat[keep],
                class_ids_flat[keep],
            ], dim=1)  # (K, 6)

            all_boxes[bi].append(dets)

    # Concatenate across scales
    result = []
    for bi in range(batch_size):
        if all_boxes[bi]:
            result.append(torch.cat(all_boxes[bi], dim=0))
        else:
            result.append(torch.zeros((0, 6), device=model_outputs[0]["cls"].device))

    return result


def batched_nms(
    detections: torch.Tensor,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
) -> torch.Tensor:
    """
    Apply Non-Maximum Suppression per class using torchvision's implementation.

    Args:
        detections   : (K, 6) tensor [x1, y1, x2, y2, conf, class_id].
        iou_threshold: IoU threshold for suppression. Default: 0.45.
        max_detections: Maximum number of detections to keep. Default: 300.

    Returns:
        Filtered detections tensor of shape (N, 6).
    """
    if detections.shape[0] == 0:
        return detections

    boxes = detections[:, :4]
    scores = detections[:, 4]
    class_ids = detections[:, 5]

    # Use abs().max() so negative decoded coordinates (boxes extending beyond
    # image edge) don't underestimate the offset and cause cross-class merging.
    offset = class_ids * (boxes.abs().max() + 1)
    boxes_offset = boxes + offset.unsqueeze(1)

    # torchvision NMS (standard, not part of our custom model)
    try:
        from torchvision.ops import nms
        keep = nms(boxes_offset, scores, iou_threshold)
    except ImportError:
        # Fallback: simple greedy NMS if torchvision unavailable
        keep = _greedy_nms(boxes_offset, scores, iou_threshold)

    keep = keep[:max_detections]
    return detections[keep]


def _greedy_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Pure-PyTorch greedy NMS (fallback if torchvision is unavailable)."""
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        idx = order[0].item()
        keep.append(idx)
        if order.numel() == 1:
            break

        rest = order[1:]
        ious = _compute_iou(boxes[idx].unsqueeze(0), boxes[rest])
        order = rest[ious.squeeze(0) <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def _compute_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """Compute IoU between one box and N boxes."""
    ax1, ay1, ax2, ay2 = box_a[0]
    bx1 = box_b[:, 0]; by1 = box_b[:, 1]; bx2 = box_b[:, 2]; by2 = box_b[:, 3]

    ix1 = torch.max(ax1, bx1)
    iy1 = torch.max(ay1, by1)
    ix2 = torch.min(ax2, bx2)
    iy2 = torch.min(ay2, by2)

    inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter + 1e-7

    return inter / union
