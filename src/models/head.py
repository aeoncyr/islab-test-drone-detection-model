"""
src/models/head.py

Decoupled Anchor-Free Detection Head (FCOS-style).

Separates classification, box regression, and objectness/centerness
into distinct branches with independent feature processing. This
decoupling prevents task interference and improves convergence speed
compared to shared-head designs, which is crucial when training from scratch.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks import ConvBNAct


class DecoupledHead(nn.Module):
    """
    FCOS-style Decoupled Detection Head.

    For each feature map level (P3, P4, P5), produces:
        - cls_out : Classification logits  (B, num_classes, H, W)
        - reg_out : Box regression offsets (B, 4, H, W)  [l, t, r, b distances]
        - obj_out : Objectness/centerness  (B, 1, H, W)

    The regression and objectness branches share their feature computation
    (reg_convs) before splitting into separate prediction layers, which
    reduces parameters while keeping spatial alignment between them.

    Classification bias is initialized with a prior probability of 0.01
    following RetinaNet's approach — this prevents the massive focal loss
    spike at step 0 that would otherwise destabilize from-scratch training.

    Args:
        num_classes    : Number of object categories. Default: 1 (drone).
        in_channels    : Input feature map channels from the neck.
        hidden_channels: Number of hidden channels in each branch. Default: 256.
    """

    def __init__(
        self,
        num_classes: int = 1,
        in_channels: int = 256,
        hidden_channels: int = 256,
    ) -> None:
        super().__init__()

        # ── Classification branch ──────────────────────────────────
        self.cls_convs = nn.Sequential(
            ConvBNAct(in_channels, hidden_channels, kernel_size=3, padding=1),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        )
        self.cls_pred = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)

        # ── Regression + Objectness branch (shared feature) ────────
        self.reg_convs = nn.Sequential(
            ConvBNAct(in_channels, hidden_channels, kernel_size=3, padding=1),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        )
        # Predicts (l, t, r, b) distances from the cell center
        self.reg_pred = nn.Conv2d(hidden_channels, 4, kernel_size=1)
        # Predicts centerness (how close the cell is to the object center)
        self.obj_pred = nn.Conv2d(hidden_channels, 1, kernel_size=1)

        self._initialize_biases(num_classes)

    def _initialize_biases(self, num_classes: int) -> None:
        """
        Initialize classification and objectness biases for prior probability = 0.01.

        Without this, at step 0 the model outputs near-50% confidence for all
        cells, creating a massive loss on all the true-negative (background) cells
        and causing gradient explosion in from-scratch training.

        Following RetinaNet (Lin et al., CVPR 2017) and FCOS, we set the bias
        so sigmoid(bias) ≈ 0.01, meaning the model starts by predicting "almost
        certainly background" everywhere — a much more stable starting point.
        """
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_value)
        nn.init.constant_(self.obj_pred.bias, bias_value)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Feature map of shape (B, in_channels, H, W).

        Returns:
            cls_out : (B, num_classes, H, W) — raw classification logits.
            reg_out : (B, 4, H, W)           — positive l,t,r,b distances.
            obj_out : (B, 1, H, W)           — centerness logits.
        """
        cls_feat = self.cls_convs(x)
        reg_feat = self.reg_convs(x)

        cls_out = self.cls_pred(cls_feat)
        # Softplus enforces non-negative l,t,r,b distances (like ReLU) but with
        # smooth, non-zero gradients everywhere — dead neurons can always recover.
        reg_out = F.softplus(self.reg_pred(reg_feat))
        obj_out = self.obj_pred(reg_feat)

        return cls_out, reg_out, obj_out
