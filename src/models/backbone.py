"""
src/models/backbone.py

Context-Enhanced Backbone for the VanillaDroneDetector.

A 4-stage, fully from-scratch CNN backbone that extracts multi-scale
feature maps at strides 8, 16, and 32. Optionally inserts CBAM
attention modules after each CSP stage (Model-C configuration).

Architecture:
    Stem      (stride 2)  → 1/2  resolution
    Stage 1   (stride 4)  → 1/4  resolution
    Stage 2   (stride 8)  → 1/8  resolution  → P3 output
    Stage 3   (stride 16) → 1/16 resolution  → P4 output
    Stage 4   (stride 32) → 1/32 resolution  → P5 output
"""

import torch
import torch.nn as nn
from typing import Tuple

from src.models.blocks import ConvBNAct, CSPBlock, CBAM


class ContextEnhancedBackbone(nn.Module):
    """
    Vanilla 4-stage backbone with optional CBAM attention.

    Outputs three feature maps (P3, P4, P5) for use by the neck.
    Kaiming Normal initialization is applied to all Conv2d layers to
    ensure stable gradient flow when training from random weights.

    Args:
        in_channels   : Number of input image channels. Default: 3 (RGB).
        base_channels : Base channel multiplier. Channels at each stage are
                        [base*2, base*4, base*8, base*16]. Default: 32.
        use_cbam      : If True, inserts a CBAM module after each CSP stage
                        (Model-C configuration). Default: False.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        use_cbam: bool = False,
        return_c2: bool = False,
    ) -> None:
        super().__init__()
        c = base_channels
        self.return_c2 = return_c2

        # ── Stem: stride 2 ─────────────────────────────────────────
        self.stem = ConvBNAct(in_channels, c, kernel_size=3, stride=2, padding=1)

        # ── Stage 1: stride 4 ──────────────────────────────────────
        self.stage1 = nn.Sequential(
            ConvBNAct(c, c * 2, kernel_size=3, stride=2, padding=1),
            CSPBlock(c * 2, c * 2, n_blocks=2),
        )
        self.cbam1 = CBAM(c * 2) if use_cbam else nn.Identity()

        # ── Stage 2: stride 8 → P3 ─────────────────────────────────
        self.stage2 = nn.Sequential(
            ConvBNAct(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
            CSPBlock(c * 4, c * 4, n_blocks=4),
        )
        self.cbam2 = CBAM(c * 4) if use_cbam else nn.Identity()

        # ── Stage 3: stride 16 → P4 ────────────────────────────────
        self.stage3 = nn.Sequential(
            ConvBNAct(c * 4, c * 8, kernel_size=3, stride=2, padding=1),
            CSPBlock(c * 8, c * 8, n_blocks=6),
        )
        self.cbam3 = CBAM(c * 8) if use_cbam else nn.Identity()

        # ── Stage 4: stride 32 → P5 ────────────────────────────────
        self.stage4 = nn.Sequential(
            ConvBNAct(c * 8, c * 16, kernel_size=3, stride=2, padding=1),
            CSPBlock(c * 16, c * 16, n_blocks=2),
        )
        self.cbam4 = CBAM(c * 16) if use_cbam else nn.Identity()

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming Normal init for Conv2d; ones/zeros for BatchNorm."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            If return_c2 is False:
                Tuple of (p3, p4, p5):
                    p3: (B, base*4,  H/8,  W/8)
                    p4: (B, base*8,  H/16, W/16)
                    p5: (B, base*16, H/32, W/32)
            If return_c2 is True:
                Tuple of (c2, p3, p4, p5):
                    c2: (B, base*2,  H/4,  W/4)
                    p3: (B, base*4,  H/8,  W/8)
                    p4: (B, base*8,  H/16, W/16)
                    p5: (B, base*16, H/32, W/32)
        """
        x = self.stem(x)
        c2 = self.cbam1(self.stage1(x))
        p3 = self.cbam2(self.stage2(c2))
        p4 = self.cbam3(self.stage3(p3))
        p5 = self.cbam4(self.stage4(p4))
        if self.return_c2:
            return c2, p3, p4, p5
        return p3, p4, p5
