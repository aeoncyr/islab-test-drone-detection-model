"""
src/engine/target_assigner.py

FCOS-style Anchor-Free Target Assigner.

Responsibility:
    Map raw ground-truth bounding boxes (in normalized YOLO format) to
    discrete target tensors at each Feature Pyramid level (P3, P4, P5).

Design decisions:
    - Scale assignment by area heuristic: small objects → P3, medium → P4, large → P5.
      Objects that don't perfectly fit a scale are still assigned to the nearest
      level to maximize positive samples from scratch.
    - Center-ness target = 1.0 for the exact center cell (simplified FCOS).
    - Kept stateless — accepts batch of boxes, returns list of target dicts.
      This allows it to be called inside the DataLoader collate_fn.
"""

from typing import Dict, List, Optional, Tuple

import torch


# Default scale assignment thresholds (by area in absolute pixels at input_size resolution)
# P2 (stride  4): objects with area <  20² px  (sub-tiny drones, <400 px²)
# P3 (stride  8): objects with area  20²–32² px (small drones, 400–1024 px²)
# P4 (stride 16): objects with area  32²–96² px (medium drones, 1024–9216 px²)
# P5 (stride 32): objects with area >  96² px  (large/close drones, >9216 px²)
_SCALE_RANGES_4LEVEL: Dict[int, Tuple[float, float]] = {
    4:  (0.0,    400.0),
    8:  (400.0,  1024.0),
    16: (1024.0, 9216.0),
    32: (9216.0, float("inf")),
}

_SCALE_RANGES_3LEVEL: Dict[int, Tuple[float, float]] = {
    8:  (0.0,    1024.0),
    16: (1024.0, 9216.0),
    32: (9216.0, float("inf")),
}


class TargetAssigner:
    """
    Assigns ground-truth boxes to feature pyramid grid cells.

    Args:
        input_size : Input image resolution (square). Default: 416.
        strides    : Pyramid level strides. Default: [8, 16, 32] or [4, 8, 16, 32].
        num_classes: Number of object classes. Default: 1.
    """

    def __init__(
        self,
        input_size: int = 416,
        strides: List[int] = None,
        num_classes: int = 1,
    ) -> None:
        self.input_size = input_size
        self.strides = strides or [8, 16, 32]
        self.num_classes = num_classes
        self.scale_ranges = _SCALE_RANGES_4LEVEL if 4 in self.strides else _SCALE_RANGES_3LEVEL

    def assign_single(
        self,
        boxes_norm: torch.Tensor,
        labels: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Assign targets for one image — fully vectorized, no Python loops.

        Args:
            boxes_norm: (N, 4) normalized [cx, cy, w, h] ground-truth boxes.
            labels    : (N,) integer class IDs.
            device    : Target device for output tensors. Defaults to boxes_norm.device.

        Returns:
            List of target dicts, one per pyramid level:
            {
                "cls_targets": (num_classes, G, G)  — one-hot class map,
                "reg_targets": (4, G, G)            — absolute (x1,y1,x2,y2),
                "obj_targets": (1, G, G)            — 1.0 at every assigned cell,
                "mask"       : (1, G, G) bool       — True where positive.
            }
        """
        if device is None:
            device = boxes_norm.device if boxes_norm.numel() > 0 else torch.device("cpu")

        # Move inputs to target device once
        boxes_norm = boxes_norm.to(device)
        labels     = labels.to(device)

        S = self.input_size
        targets = []

        for stride in self.strides:
            G = S // stride

            # Allocate target tensors directly on device — no CPU→GPU copy later
            cls_t = torch.zeros((self.num_classes, G, G), device=device)
            reg_t = torch.zeros((4, G, G), device=device)
            obj_t = torch.zeros((1, G, G), device=device)
            mask  = torch.zeros((1, G, G), dtype=torch.bool, device=device)

            N = boxes_norm.shape[0]
            if N > 0:
                # ── Convert normalized → absolute pixel coords ──────────────
                abs_cx = boxes_norm[:, 0] * S          # (N,)
                abs_cy = boxes_norm[:, 1] * S
                abs_w  = boxes_norm[:, 2] * S
                abs_h  = boxes_norm[:, 3] * S
                x1 = abs_cx - abs_w / 2.0
                y1 = abs_cy - abs_h / 2.0
                x2 = abs_cx + abs_w / 2.0
                y2 = abs_cy + abs_h / 2.0
                area = abs_w * abs_h                    # (N,)

                # ── Vectorized scale-range filter ───────────────────────────
                lo, hi = self.scale_ranges[stride]
                in_range = (area >= lo) & (area < hi)  # (N,)

                # Fallback: boxes not claimed by ANY level get assigned here
                if not in_range.all():
                    claimed = torch.zeros(N, dtype=torch.bool, device=device)
                    for s in self.strides:
                        lo_s, hi_s = self.scale_ranges[s]
                        claimed |= (area >= lo_s) & (area < hi_s)
                    in_range = in_range | ~claimed

                valid_idx = in_range.nonzero(as_tuple=False).squeeze(-1)  # (V,)

                if valid_idx.numel() > 0:
                    bx1   = x1[valid_idx]       # (V,)
                    by1   = y1[valid_idx]
                    bx2   = x2[valid_idx]
                    by2   = y2[valid_idx]
                    b_area = area[valid_idx]     # (V,) — for tie-breaking
                    b_cls  = labels[valid_idx].clamp(0, self.num_classes - 1).long()  # (V,)

                    # ── Build cell-center grid ──────────────────────────────
                    yi, xi = torch.meshgrid(
                        torch.arange(G, device=device, dtype=torch.float32),
                        torch.arange(G, device=device, dtype=torch.float32),
                        indexing="ij",
                    )
                    cx_cells = (xi + 0.5) * stride  # (G, G)
                    cy_cells = (yi + 0.5) * stride

                    # ── Multi-cell FCOS center-region assignment ────────────
                    # (V, G, G): True when cell center falls inside the GT box
                    cell_in_box = (
                        (cx_cells.unsqueeze(0) >= bx1.view(-1, 1, 1)) &
                        (cx_cells.unsqueeze(0) <  bx2.view(-1, 1, 1)) &
                        (cy_cells.unsqueeze(0) >= by1.view(-1, 1, 1)) &
                        (cy_cells.unsqueeze(0) <  by2.view(-1, 1, 1))
                    )  # (V, G, G)

                    # Guarantee center cell is always assigned (handles tiny
                    # drones smaller than a single grid cell)
                    gcx_c = abs_cx[valid_idx].div(stride, rounding_mode="floor").long().clamp(0, G - 1)
                    gcy_c = abs_cy[valid_idx].div(stride, rounding_mode="floor").long().clamp(0, G - 1)
                    v_idx = torch.arange(valid_idx.numel(), device=device)
                    cell_in_box[v_idx, gcy_c, gcx_c] = True

                    # ── Tie-breaking: smallest-area box wins ────────────────
                    # Where no box covers a cell → inf so argmin ignores it
                    area_map = b_area.view(-1, 1, 1).expand_as(cell_in_box)
                    area_map = torch.where(cell_in_box, area_map,
                                           torch.full_like(area_map, float("inf")))
                    min_area, winner = area_map.min(dim=0)  # (G, G)

                    assigned = min_area < float("inf")     # (G, G) — cells with a winner
                    if assigned.any():
                        gy, gx = assigned.nonzero(as_tuple=True)  # (M,)
                        w_idx  = winner[gy, gx]                   # (M,) index into V

                        mask[0, gy, gx]             = True
                        cls_t[b_cls[w_idx], gy, gx] = 1.0
                        reg_t[0, gy, gx] = bx1[w_idx]
                        reg_t[1, gy, gx] = by1[w_idx]
                        reg_t[2, gy, gx] = bx2[w_idx]
                        reg_t[3, gy, gx] = by2[w_idx]
                        obj_t[0, gy, gx] = 1.0

            targets.append({
                "cls_targets": cls_t,
                "reg_targets": reg_t,
                "obj_targets": obj_t,
                "mask":        mask,
            })

        return targets

    def collate_fn(
        self,
        batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Lightweight collate: stacks images only. Raw boxes/labels are kept
        as lists so workers do zero tensor allocation beyond image I/O.

        Use this as the DataLoader `collate_fn`; call `assign_batch` in the
        training loop (after `.to(device)`) to run target assignment on GPU.

        Returns:
            images    : (B, 3, H, W) stacked image tensor.
            all_boxes : List[B] of (N_i, 4) box tensors (normalized cx,cy,w,h).
            all_labels: List[B] of (N_i,) label tensors.
        """
        images, all_boxes, all_labels = zip(*batch)
        return torch.stack(images, dim=0), list(all_boxes), list(all_labels)

    def assign_batch(
        self,
        all_boxes: List[torch.Tensor],
        all_labels: List[torch.Tensor],
        device: torch.device,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Run target assignment for a full batch on the given device.

        Call this in the training loop after moving images to `device`:

            images = images.to(device)
            batch_targets = assigner.assign_batch(all_boxes, all_labels, device)

        Returns:
            batch_targets: List of 3 level-dicts with (B,...) tensors on `device`.
        """
        per_image_targets = [
            self.assign_single(boxes, labels, device=device)
            for boxes, labels in zip(all_boxes, all_labels)
        ]

        batch_targets = []
        for level_idx in range(len(self.strides)):
            batch_targets.append({
                "cls_targets": torch.stack([t[level_idx]["cls_targets"] for t in per_image_targets]),
                "reg_targets": torch.stack([t[level_idx]["reg_targets"] for t in per_image_targets]),
                "obj_targets": torch.stack([t[level_idx]["obj_targets"] for t in per_image_targets]),
                "mask":        torch.stack([t[level_idx]["mask"]        for t in per_image_targets]),
            })
        return batch_targets

    def collate_and_assign(
        self,
        batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]], List[torch.Tensor], List[torch.Tensor]]:
        """
        Legacy collate_fn that also does target assignment in the worker.
        Kept for backward compatibility — prefer collate_fn + assign_batch.
        """
        images, all_boxes, all_labels = zip(*batch)
        images = torch.stack(images, dim=0)

        per_image_targets = [
            self.assign_single(boxes, labels)
            for boxes, labels in zip(all_boxes, all_labels)
        ]

        batch_targets = []
        for level_idx in range(len(self.strides)):
            batch_targets.append({
                "cls_targets": torch.stack([t[level_idx]["cls_targets"] for t in per_image_targets]),
                "reg_targets": torch.stack([t[level_idx]["reg_targets"] for t in per_image_targets]),
                "obj_targets": torch.stack([t[level_idx]["obj_targets"] for t in per_image_targets]),
                "mask":        torch.stack([t[level_idx]["mask"]        for t in per_image_targets]),
            })

        return images, batch_targets, list(all_boxes), list(all_labels)
