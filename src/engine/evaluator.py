"""
src/engine/evaluator.py

mAP Evaluator for the VanillaDroneDetector.

Computes standard COCO-style detection metrics:
    - mAP@50    : Mean Average Precision at IoU threshold 0.50.
    - mAP@50:95 : Mean AP averaged over IoU thresholds 0.50:0.05:0.95.
    - Precision-Recall curve per class.

Implementation:
    - Predictions are post-processed with confidence thresholding + NMS.
    - TP/FP/FN are accumulated across the validation set.
    - AP is computed via the 11-point interpolation method (standard for VOC-style).
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils.box_ops import decode_predictions, batched_nms


class Evaluator:
    """
    Computes mAP@50, mAP@50:95, latency (ms), and FPS on a validation DataLoader.

    Args:
        model          : The detector model (set to eval mode internally).
        val_loader     : Validation DataLoader.
        num_classes    : Number of object classes. Default: 1.
        conf_threshold : Minimum confidence for predictions. Default: 0.05.
        nms_threshold  : IoU threshold for NMS. Default: 0.45.
        device         : Computation device.
    """

    _IOU_THRESHOLDS = np.arange(0.50, 1.00, 0.05)  # [0.50, 0.55, ..., 0.95]

    def __init__(
        self,
        model: nn.Module,
        val_loader,
        num_classes: int = 1,
        conf_threshold: float = 0.05,
        nms_threshold: float = 0.45,
        device: Optional[torch.device] = None,
        assigner=None,
    ) -> None:
        self.model = model
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.assigner = assigner

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Run full evaluation pass and compute mAP metrics alongside latency and FPS.

        Returns:
            Dict with keys: "mAP50", "mAP50_95", "precision", "recall", "latency_ms", "fps".
        """
        self.model.eval()

        all_predictions: List[torch.Tensor] = []
        all_ground_truths: List[torch.Tensor] = []

        total_inf_time = 0.0
        total_images = 0

        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
            images, batch_targets, all_boxes, all_labels = batch
            images = images.to(self.device, non_blocking=True).float() / 255.0
            input_size = images.shape[-1]
            b_size = images.shape[0]

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            outputs = self.model(images)
            decoded = decode_predictions(outputs, conf_threshold=self.conf_threshold)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            total_inf_time += (time.perf_counter() - t0)
            total_images += b_size

            for bi, dets in enumerate(decoded):
                dets = dets.to("cpu")
                all_predictions.append(batched_nms(dets, iou_threshold=self.nms_threshold))

            # Build GT directly from raw normalized boxes
            for bi in range(images.shape[0]):
                boxes_bi  = all_boxes[bi]
                labels_bi = all_labels[bi]
                if boxes_bi.shape[0] > 0:
                    cx = boxes_bi[:, 0] * input_size
                    cy = boxes_bi[:, 1] * input_size
                    w  = boxes_bi[:, 2] * input_size
                    h  = boxes_bi[:, 3] * input_size
                    x1 = torch.clamp(cx - w / 2.0, min=0.0)
                    y1 = torch.clamp(cy - h / 2.0, min=0.0)
                    x2 = torch.clamp(cx + w / 2.0, max=float(input_size))
                    y2 = torch.clamp(cy + h / 2.0, max=float(input_size))
                    gt = torch.stack([x1, y1, x2, y2, labels_bi.float()], dim=1)
                else:
                    gt = torch.zeros((0, 5), dtype=torch.float32)
                all_ground_truths.append(gt)

        metrics = self._compute_map(all_predictions, all_ground_truths)
        metrics["latency_ms"] = (total_inf_time / total_images * 1000.0) if total_images > 0 else 0.0
        metrics["fps"] = (total_images / total_inf_time) if total_inf_time > 0 else 0.0

        return metrics


    def _collect_gt_boxes(
        self,
        batch_targets: List[Dict[str, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Reconstruct absolute GT boxes from target tensors for one image.

        Returns:
            Tensor (M, 5): [x1, y1, x2, y2, class_id].
        """
        gt_list = []
        for level_target in batch_targets:
            mask = level_target["mask"][batch_idx, 0]   # (G, G)
            reg_t = level_target["reg_targets"][batch_idx]  # (4, G, G)
            cls_t = level_target["cls_targets"][batch_idx]  # (C, G, G)

            pos_indices = mask.nonzero(as_tuple=False)  # (K, 2): [row, col]
            for yx in pos_indices:
                y, x = yx[0].item(), yx[1].item()
                box = reg_t[:, y, x].tolist()  # [x1, y1, x2, y2]
                class_id = cls_t[:, y, x].argmax().item()
                gt_list.append(box + [float(class_id)])

        if gt_list:
            return torch.tensor(gt_list, dtype=torch.float32)
        return torch.zeros((0, 5), dtype=torch.float32)

    def _compute_map(
        self,
        all_predictions: List[torch.Tensor],
        all_ground_truths: List[torch.Tensor],
    ) -> Dict[str, float]:
        """
        Compute mAP@50 and mAP@50:95 across all images.
        Uses 11-point interpolation for AP computation.
        """
        ap50_list = []
        ap50_95_list = []

        for cls_id in range(self.num_classes):
            # Collect class-specific predictions and GTs
            cls_preds = []
            cls_gt_counts = []

            for img_idx, (preds, gts) in enumerate(zip(all_predictions, all_ground_truths)):
                # Filter predictions for this class
                if preds.shape[0] > 0:
                    cls_mask = preds[:, 5].long() == cls_id
                    cls_preds.append((img_idx, preds[cls_mask]))
                else:
                    cls_preds.append((img_idx, preds))

                # Count GT boxes for this class in this image
                if gts.shape[0] > 0:
                    gt_cls_mask = gts[:, 4].long() == cls_id
                    cls_gt_counts.append(int(gt_cls_mask.sum().item()))
                else:
                    cls_gt_counts.append(0)

            total_gt = sum(cls_gt_counts)
            if total_gt == 0:
                continue

            aps_at_thresholds = []
            for iou_thresh in self._IOU_THRESHOLDS:
                ap = self._compute_ap_at_threshold(
                    cls_preds, all_ground_truths, cls_id, iou_thresh, total_gt
                )
                aps_at_thresholds.append(ap)

            ap50_list.append(aps_at_thresholds[0])
            ap50_95_list.append(np.mean(aps_at_thresholds))

        mAP50 = float(np.mean(ap50_list)) if ap50_list else 0.0
        mAP50_95 = float(np.mean(ap50_95_list)) if ap50_95_list else 0.0

        # Compute overall precision/recall at 0.5 threshold for last class processed
        precision, recall = self._compute_pr_at_50(
            all_predictions, all_ground_truths, cls_id=0
        )

        return {
            "mAP50": mAP50,
            "mAP50_95": mAP50_95,
            "precision": precision,
            "recall": recall,
        }

    def _compute_ap_at_threshold(
        self,
        cls_preds: list,
        all_gts: List[torch.Tensor],
        cls_id: int,
        iou_threshold: float,
        total_gt: int,
    ) -> float:
        """Compute AP for a single class at a given IoU threshold."""
        # Collect all detections sorted by confidence
        det_records = []
        for img_idx, preds in cls_preds:
            if preds.shape[0] == 0:
                continue
            for pred in preds:
                det_records.append({
                    "img_idx": img_idx,
                    "box": pred[:4].numpy(),
                    "conf": pred[4].item(),
                })

        if not det_records:
            return 0.0

        det_records.sort(key=lambda d: -d["conf"])

        tp = np.zeros(len(det_records))
        fp = np.zeros(len(det_records))
        matched_gt: Dict[int, set] = {i: set() for i in range(len(all_gts))}

        for di, det in enumerate(det_records):
            img_idx = det["img_idx"]
            gts = all_gts[img_idx]

            if gts.shape[0] == 0:
                fp[di] = 1
                continue

            # Filter GT for this class
            gt_cls_mask = gts[:, 4].long() == cls_id
            gt_boxes = gts[gt_cls_mask, :4]

            if gt_boxes.shape[0] == 0:
                fp[di] = 1
                continue

            ious = self._iou_matrix(
                torch.tensor(det["box"]).unsqueeze(0),
                gt_boxes,
            ).squeeze(0)  # (num_gt,)

            if ious.numel() == 0:
                fp[di] = 1
                continue

            best_iou, best_gt_idx = ious.max(dim=0)
            best_iou = best_iou.item()
            best_gt_idx = best_gt_idx.item()

            if best_iou >= iou_threshold and best_gt_idx not in matched_gt[img_idx]:
                tp[di] = 1
                matched_gt[img_idx].add(best_gt_idx)
            else:
                fp[di] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / (total_gt + 1e-7)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)

        return self._ap_from_pr(precisions, recalls)

    def _compute_pr_at_50(
        self,
        all_preds: List[torch.Tensor],
        all_gts: List[torch.Tensor],
        cls_id: int,
    ) -> Tuple[float, float]:
        """Quick precision/recall summary at IoU=0.50."""
        tp_total = fp_total = fn_total = 0
        for preds, gts in zip(all_preds, all_gts):
            if gts.shape[0] > 0:
                gt_cls = gts[gts[:, 4].long() == cls_id, :4]
            else:
                gt_cls = gts[:, :4]

            if preds.shape[0] > 0:
                p_cls = preds[preds[:, 5].long() == cls_id, :4]
            else:
                p_cls = preds[:, :4]

            matched = 0
            if p_cls.shape[0] > 0 and gt_cls.shape[0] > 0:
                ious = self._iou_matrix(p_cls, gt_cls)
                matched = int((ious.max(dim=1).values >= 0.5).sum().item())

            tp_total += matched
            fp_total += max(p_cls.shape[0] - matched, 0)
            fn_total += max(gt_cls.shape[0] - matched, 0)

        precision = tp_total / (tp_total + fp_total + 1e-7)
        recall = tp_total / (tp_total + fn_total + 1e-7)
        return float(precision), float(recall)

    @staticmethod
    def _ap_from_pr(precisions: np.ndarray, recalls: np.ndarray) -> float:
        """11-point interpolated Average Precision."""
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            prec_at_rec = precisions[recalls >= t]
            if len(prec_at_rec) == 0:
                p = 0.0
            else:
                p = np.max(prec_at_rec)
            ap += p / 11.0
        return float(ap)

    @staticmethod
    def _iou_matrix(
        boxes_a: torch.Tensor,
        boxes_b: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise IoU matrix between two sets of boxes (x1,y1,x2,y2)."""
        ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
        bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

        inter_x1 = torch.max(ax1.unsqueeze(1), bx1.unsqueeze(0))
        inter_y1 = torch.max(ay1.unsqueeze(1), by1.unsqueeze(0))
        inter_x2 = torch.min(ax2.unsqueeze(1), bx2.unsqueeze(0))
        inter_y2 = torch.min(ay2.unsqueeze(1), by2.unsqueeze(0))

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        area_a = ((ax2 - ax1) * (ay2 - ay1)).unsqueeze(1)
        area_b = ((bx2 - bx1) * (by2 - by1)).unsqueeze(0)
        union = area_a + area_b - inter_area + 1e-7

        return inter_area / union
