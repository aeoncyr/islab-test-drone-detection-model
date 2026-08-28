"""
src/losses/multi_task_losses.py

Custom Multi-Task Loss for the VanillaDroneDetector.

All loss functions are hand-crafted — no external loss libraries used.
Standard components (BCE, Cross-Entropy, MSE/MAE) are used as building
blocks where the task permits, in compliance with assignment requirements.

Loss Components:
    1. FocalLoss         : Classification — down-weights easy negatives.
    2. CIoULoss          : Regression    — penalizes center distance & aspect ratio.
    3. NWDLoss           : Regression    — Wasserstein metric for tiny/no-overlap boxes.
    4. HybridMultiTaskLoss: Orchestrates all sub-losses across P3/P4/P5.

The Hybrid Regression Loss linearly combines CIoU and NWD:
    L_reg = α * L_CIoU + (1 - α) * L_NWD
This combination is essential for drone detection: CIoU handles larger,
overlapping boxes well while NWD remains effective for tiny objects where
IoU collapses to near-zero.
"""

import math
import warnings
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Box Decoding Utility
# ==============================================================================

def decode_boxes_from_ltrb(reg_preds: torch.Tensor, stride: int) -> torch.Tensor:
    """
    Convert network distance predictions (l, t, r, b) to absolute bounding
    box coordinates (x1, y1, x2, y2) using the grid cell centers.

    Each grid cell at position (i, j) has its center at:
        x_center = (j + 0.5) * stride
        y_center = (i + 0.5) * stride

    The predicted distances define:
        x1 = x_center - l,  y1 = y_center - t
        x2 = x_center + r,  y2 = y_center + b

    Args:
        reg_preds: Tensor of shape (B, 4, H, W) — [l, t, r, b] distances.
        stride   : Feature map downsampling stride (8, 16, or 32).

    Returns:
        Decoded boxes of shape (B, 4, H, W) — [x1, y1, x2, y2].
    """
    b, _, h, w = reg_preds.shape
    device = reg_preds.device

    y_grid, x_grid = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )

    x_centers = (x_grid.float() + 0.5) * stride  # (H, W)
    y_centers = (y_grid.float() + 0.5) * stride  # (H, W)

    # Expand to (B, 1, H, W) for broadcasting
    x_centers = x_centers.unsqueeze(0).unsqueeze(0).expand(b, 1, h, w)
    y_centers = y_centers.unsqueeze(0).unsqueeze(0).expand(b, 1, h, w)

    l, t, r, bot = reg_preds.split(1, dim=1)

    x1 = x_centers - l
    y1 = y_centers - t
    x2 = x_centers + r
    y2 = y_centers + bot

    return torch.cat([x1, y1, x2, y2], dim=1)


def bbox_to_cxcywh(bboxes: torch.Tensor) -> torch.Tensor:
    """Convert (x1, y1, x2, y2) to (cx, cy, w, h)."""
    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    return torch.stack([cx, cy, w, h], dim=1)


# ==============================================================================
# 1. Focal Loss (Classification)
# ==============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection (Lin et al., CVPR 2017).

    Addresses the extreme class imbalance in anchor-free detection by
    down-weighting well-classified background cells (easy negatives) and
    focusing training on hard, misclassified foreground examples (drones).

    Loss formula:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Where:
        p_t = sigmoid(logit) for positive class,  1 - sigmoid for negative.
        α   = class balance weight (0.25 for positive class).
        γ   = focusing parameter (γ=2 reduces easy-example loss by ~100×).

    Args:
        alpha    : Positive-class weighting factor. Default: 0.25.
        gamma    : Focusing exponent. Default: 2.0.
        reduction: Aggregation method ("sum" | "mean"). Default: "sum".
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "sum",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs : Raw logits, shape (N, C).
            targets: Binary ground-truth labels, shape (N, C).

        Returns:
            Scalar focal loss.
        """
        # Binary cross-entropy is the baseline (standard allowed loss)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)  # Probability of the true class

        # Class-conditional alpha: α for positives (y=1), (1-α) for negatives (y=0)
        # This matches the original RetinaNet paper (Lin et al., CVPR 2017)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal_weight = alpha_t * (1.0 - pt) ** self.gamma
        loss = focal_weight * bce_loss

        return loss.sum() if self.reduction == "sum" else loss.mean()

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, gamma={self.gamma}, reduction={self.reduction}"


# ==============================================================================
# 2. CIoU Loss (Box Regression)
# ==============================================================================

class CIoULoss(nn.Module):
    """
    Complete Intersection over Union Loss (Zheng et al., AAAI 2020).

    Extends the standard IoU loss with three penalty terms:
        1. Non-overlap: IoU numerically handles non-overlapping boxes.
        2. Center distance: ρ²(center_pred, center_gt) / c²
        3. Aspect ratio consistency: α * v  where v = (4/π²)(arctan(w_gt/h_gt) - arctan(w/h))²

    CIoU provides richer regression supervision than MSE/MAE especially
    for overlapping boxes, while the aspect-ratio term encourages
    the predicted box to match the drone's true shape.

    Args:
        eps: Numerical stability epsilon. Default: 1e-7.
    """

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_boxes  : (N, 4) predicted boxes in (x1, y1, x2, y2).
            target_boxes: (N, 4) ground-truth boxes in (x1, y1, x2, y2).

        Returns:
            Per-box CIoU loss tensor of shape (N,).
        """
        px1, py1, px2, py2 = pred_boxes.unbind(dim=1)
        gx1, gy1, gx2, gy2 = target_boxes.unbind(dim=1)

        # ── Intersection ──────────────────────────────────────────
        ix1 = torch.max(px1, gx1)
        iy1 = torch.max(py1, gy1)
        ix2 = torch.min(px2, gx2)
        iy2 = torch.min(py2, gy2)
        inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)

        # ── Union & IoU ────────────────────────────────────────────
        pred_area = torch.clamp(px2 - px1, min=0) * torch.clamp(py2 - py1, min=0)
        gt_area = torch.clamp(gx2 - gx1, min=0) * torch.clamp(gy2 - gy1, min=0)
        union = pred_area + gt_area - inter + self.eps
        iou = inter / union

        # ── Center distance penalty ────────────────────────────────
        enc_x1 = torch.min(px1, gx1)
        enc_y1 = torch.min(py1, gy1)
        enc_x2 = torch.max(px2, gx2)
        enc_y2 = torch.max(py2, gy2)
        c_diag_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + self.eps

        pred_cx, pred_cy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
        gt_cx, gt_cy = (gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0
        center_dist_sq = (pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2
        rho_penalty = center_dist_sq / c_diag_sq

        # ── Aspect ratio penalty ───────────────────────────────────
        pred_w = torch.clamp(px2 - px1, min=self.eps)
        pred_h = torch.clamp(py2 - py1, min=self.eps)
        gt_w = torch.clamp(gx2 - gx1, min=self.eps)
        gt_h = torch.clamp(gy2 - gy1, min=self.eps)

        v = (4.0 / (math.pi ** 2)) * (torch.atan(gt_w / gt_h) - torch.atan(pred_w / pred_h)) ** 2

        with torch.no_grad():
            alpha_weight = v / (1.0 - iou + v + self.eps)

        ciou = iou - rho_penalty - alpha_weight * v
        # Clamp to prevent extreme loss values from degenerate predictions
        return torch.clamp(1.0 - ciou, min=0.0, max=4.0)


# ==============================================================================
# 3. NWD Loss (Regression for Tiny Objects)
# ==============================================================================

class NWDLoss(nn.Module):
    """
    Normalized Wasserstein Distance Loss (Wang et al., 2021).

    Models each bounding box as a 2D Gaussian distribution N(μ, Σ):
        μ = (cx, cy)
        Σ = diag(w/2, h/2)²

    The 2nd-order Wasserstein Distance between two Gaussians has a
    closed-form solution:
        W₂² = ||μ_pred - μ_gt||² + ||σ_pred - σ_gt||²_F

    This is then normalized via an exponential into a similarity in [0, 1]:
        NWD = exp(-√W₂² / C)

    NWD remains effective when IoU ≈ 0 (tiny drones, no-overlap predictions),
    which is a critical failure mode for pure CIoU-based training.

    Args:
        constant: Normalization constant C (dataset-specific).
                  12.8 is empirically effective for UAV/tiny objects.
    """

    def __init__(self, constant: float = 12.8) -> None:
        super().__init__()
        self.C = constant

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_boxes  : (N, 4) predicted boxes in (x1, y1, x2, y2).
            target_boxes: (N, 4) ground-truth boxes in (x1, y1, x2, y2).

        Returns:
            Per-box NWD loss tensor of shape (N,).
        """
        p = bbox_to_cxcywh(pred_boxes)
        t = bbox_to_cxcywh(target_boxes)

        # Gaussian center distance
        center_dist_sq = (p[:, 0] - t[:, 0]) ** 2 + (p[:, 1] - t[:, 1]) ** 2

        # Gaussian shape distance (simplified Frobenius norm for axis-aligned boxes)
        shape_dist_sq = ((p[:, 2] - t[:, 2]) ** 2 + (p[:, 3] - t[:, 3]) ** 2) / 4.0

        w2_sq = center_dist_sq + shape_dist_sq
        nwd = torch.exp(-torch.sqrt(w2_sq + 1e-7) / self.C)

        return 1.0 - nwd


# ==============================================================================
# 4. Hybrid Multi-Task Loss Orchestrator
# ==============================================================================

class HybridMultiTaskLoss(nn.Module):
    """
    End-to-End Loss Orchestrator for the VanillaDroneDetector.

    Manages all three detection sub-tasks across all pyramid levels:
        L_total = λ_cls * L_focal + λ_obj * L_bce_obj + λ_reg * L_hybrid_reg

    Where:
        L_hybrid_reg = α * L_CIoU + (1 - α) * L_NWD

    All losses are normalized by the total number of positive (foreground)
    assignments across all scales to prevent scale-dependent magnitude issues.

    Args:
        num_classes  : Number of object classes.
        lambda_cls   : Weight for classification loss. Default: 1.0.
        lambda_reg   : Weight for regression loss. Default: 2.0.
        lambda_obj   : Weight for objectness loss. Default: 1.0.
        alpha_hybrid : Blend weight between CIoU and NWD. Default: 0.5.
        focal_alpha  : Focal Loss alpha. Default: 0.25.
        focal_gamma  : Focal Loss gamma. Default: 2.0.
        nwd_constant : NWD normalization constant. Default: 12.8.
    """

    def __init__(
        self,
        num_classes: int = 1,
        lambda_cls: float = 1.0,
        lambda_reg: float = 2.0,
        lambda_obj: float = 1.0,
        alpha_hybrid: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        nwd_constant: float = 12.8,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg
        self.lambda_obj = lambda_obj
        self.alpha_hybrid = alpha_hybrid

        self.cls_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.ciou_loss_fn = CIoULoss()
        self.nwd_loss_fn = NWDLoss(constant=nwd_constant)
        # BCEWithLogitsLoss with reduction='mean' normalizes over ALL cells
        # (positives + negatives), keeping objectness loss on the same scale
        # as classification and regression losses. Using 'sum'/num_pos would
        # amplify it by ~700× (57,344 cells / ~80 positives) and dominate training.
        self.obj_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(
        self,
        model_outputs: List[Dict[str, Any]],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total multi-task loss.

        Args:
            model_outputs: List of 3 dicts (P3, P4, P5) from the detector forward pass.
                           Each dict has keys: "stride", "cls", "reg", "obj".
            targets      : List of 3 dicts with pre-assigned targets per pyramid level.
                           Keys: "cls_targets", "reg_targets", "obj_targets", "mask".

        Returns:
            Tuple of:
                total_loss : Scalar combined loss tensor.
                loss_dict  : {"loss_cls", "loss_obj", "loss_reg"} float breakdown.
        """
        loss_cls = torch.tensor(0.0, device=model_outputs[0]["cls"].device)
        loss_reg = torch.tensor(0.0, device=model_outputs[0]["cls"].device)
        loss_obj = torch.tensor(0.0, device=model_outputs[0]["cls"].device)
        num_pos_total = 0

        _DEFAULT_STRIDES = [8, 16, 32]
        for i, (pred, target) in enumerate(zip(model_outputs, targets)):
            if "stride" in pred:
                s = pred["stride"]
                stride = int(s.flatten()[0].item()) if isinstance(s, torch.Tensor) else int(s)
            else:
                stride = _DEFAULT_STRIDES[i]

            # Flatten spatial dims: (B, C, H, W) → (B*H*W, C)
            cls_pred = pred["cls"].permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            reg_pred = pred["reg"].permute(0, 2, 3, 1).reshape(-1, 4)
            obj_pred = pred["obj"].permute(0, 2, 3, 1).reshape(-1)

            cls_t = target["cls_targets"].permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            reg_t = target["reg_targets"].permute(0, 2, 3, 1).reshape(-1, 4)
            obj_t = target["obj_targets"].permute(0, 2, 3, 1).reshape(-1)
            pos_mask = target["mask"].permute(0, 2, 3, 1).reshape(-1)

            num_pos = int(pos_mask.sum().item())
            num_pos_total += num_pos

            # ── A. Classification Loss ─────────────────────────────
            loss_cls = loss_cls + self.cls_loss_fn(cls_pred, cls_t)

            # ── B. Objectness Loss ─────────────────────────────────
            loss_obj = loss_obj + self.obj_loss_fn(obj_pred, obj_t)

            # ── C. Regression Loss (positive samples only) ─────────
            if num_pos > 0:
                # Decode distance predictions to absolute (x1, y1, x2, y2)
                decoded_boxes = decode_boxes_from_ltrb(pred["reg"], stride)
                decoded_flat = decoded_boxes.permute(0, 2, 3, 1).reshape(-1, 4)

                pos_pred_boxes = decoded_flat[pos_mask]
                pos_target_boxes = reg_t[pos_mask]

                l_ciou = self.ciou_loss_fn(pos_pred_boxes, pos_target_boxes)
                l_nwd = self.nwd_loss_fn(pos_pred_boxes, pos_target_boxes)

                # Hybrid combination: weighted sum of CIoU and NWD per box
                hybrid = self.alpha_hybrid * l_ciou + (1.0 - self.alpha_hybrid) * l_nwd
                loss_reg = loss_reg + hybrid.sum()

        # Normalize cls and reg losses by positive count.
        # obj loss already averaged over all cells by BCEWithLogitsLoss(reduction='mean'),
        # so it only needs the task weight — not division by num_pos.
        normalizer = max(num_pos_total, 1)

        loss_cls = loss_cls / normalizer * self.lambda_cls
        loss_obj = loss_obj * self.lambda_obj          # mean over cells, not pos-normalized
        loss_reg = loss_reg / normalizer * self.lambda_reg

        total_loss = loss_cls + loss_obj + loss_reg

        # Guard against NaN/Inf propagating through the network
        if not torch.isfinite(total_loss):
            warnings.warn(
                f"[Loss] Non-finite total_loss detected "
                f"(cls={loss_cls.item():.4f}, obj={loss_obj.item():.4f}, "
                f"reg={loss_reg.item():.4f}). Clamping to 0.",
                RuntimeWarning,
                stacklevel=2,
            )
            total_loss = torch.zeros_like(total_loss)

        loss_dict = {
            "loss_cls": loss_cls.item(),
            "loss_obj": loss_obj.item(),
            "loss_reg": loss_reg.item(),
        }
        return total_loss, loss_dict
