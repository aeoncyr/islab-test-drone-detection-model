"""
src/models/neck.py

Feature Pyramid Network variants for the VanillaDroneDetector.

Two neck implementations are provided:
  - FPN      : Top-down only (Model-A baseline).
  - FPNPAN   : FPN + bottom-up PAN path (Model-B and Model-C).
               The bidirectional design helps preserve fine-grained
               spatial detail for small UAV objects at P3.

References:
    FPN  — Lin et al., CVPR 2017
    PAN  — Liu et al., CVPR 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from src.models.blocks import ConvBNAct, CSPBlock


class FPN(nn.Module):
    """
    Standard Feature Pyramid Network (top-down only).

    Fuses high-level semantic information from P5 downward into P3.
    Each level is reduced to `out_channels` via lateral 1×1 convolutions
    followed by a 3×3 anti-aliasing blend convolution.

    Args:
        in_channels_list : Channel counts for [P3, P4, P5] from the backbone.
        out_channels     : Uniform output channel count for all pyramid levels.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
    ) -> None:
        super().__init__()
        c3_in, c4_in, c5_in = in_channels_list

        # Lateral 1×1 projections
        self.lat_c5 = ConvBNAct(c5_in, out_channels, kernel_size=1, padding=0)
        self.lat_c4 = ConvBNAct(c4_in, out_channels, kernel_size=1, padding=0)
        self.lat_c3 = ConvBNAct(c3_in, out_channels, kernel_size=1, padding=0)

        # Post-merge 3×3 blend (removes upsampling aliasing)
        self.blend_p4 = ConvBNAct(out_channels, out_channels, kernel_size=3, padding=1)
        self.blend_p3 = ConvBNAct(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c3, c4, c5 = features

        # Top-down path
        p5 = self.lat_c5(c5)

        p5_up = F.interpolate(p5, size=c4.shape[2:], mode="nearest")
        p4 = self.blend_p4(self.lat_c4(c4) + p5_up)

        p4_up = F.interpolate(p4, size=c3.shape[2:], mode="nearest")
        p3 = self.blend_p3(self.lat_c3(c3) + p4_up)

        return p3, p4, p5


class FPNPAN(nn.Module):
    """
    Feature Pyramid Network with Path Aggregation Network (FPN + PAN).

    Extends FPN with a bottom-up path that re-propagates fine-grained
    spatial features from P3 back up to P4 and P5. The second pass
    allows low-level details (texture, edges) to reach higher levels,
    which is especially beneficial for detecting small drones.

    Architecture:
        Top-down  (FPN):  P5 → P4 → P3   (semantic enrichment)
        Bottom-up (PAN):  P3 → P4 → P5   (spatial detail propagation)

    Args:
        in_channels_list : Channel counts for [P3, P4, P5] from the backbone.
        out_channels     : Uniform output channel count for all pyramid levels.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
    ) -> None:
        super().__init__()
        c3_in, c4_in, c5_in = in_channels_list

        # ── Top-down (FPN) path ────────────────────────────────────
        self.lat_c5 = ConvBNAct(c5_in, out_channels, kernel_size=1, padding=0)
        self.lat_c4 = ConvBNAct(c4_in, out_channels, kernel_size=1, padding=0)
        self.lat_c3 = ConvBNAct(c3_in, out_channels, kernel_size=1, padding=0)

        self.td_blend_p4 = CSPBlock(out_channels, out_channels, n_blocks=1)
        self.td_blend_p3 = CSPBlock(out_channels, out_channels, n_blocks=1)

        # ── Bottom-up (PAN) path ───────────────────────────────────
        # Downsampling convolutions for the upward path
        self.bu_down_p3 = ConvBNAct(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bu_down_p4 = ConvBNAct(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

        # Fusion CSP blocks after merging with top-down features
        self.bu_blend_p4 = CSPBlock(out_channels * 2, out_channels, n_blocks=1)
        self.bu_blend_p5 = CSPBlock(out_channels * 2, out_channels, n_blocks=1)

    def forward(
        self,
        features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c3, c4, c5 = features

        # ── Top-down FPN ───────────────────────────────────────────
        td_p5 = self.lat_c5(c5)

        td_p5_up = F.interpolate(td_p5, size=c4.shape[2:], mode="nearest")
        td_p4 = self.td_blend_p4(self.lat_c4(c4) + td_p5_up)

        td_p4_up = F.interpolate(td_p4, size=c3.shape[2:], mode="nearest")
        td_p3 = self.td_blend_p3(self.lat_c3(c3) + td_p4_up)

        # ── Bottom-up PAN ──────────────────────────────────────────
        bu_p3 = td_p3  # Already enriched by FPN

        bu_p3_down = self.bu_down_p3(bu_p3)
        bu_p4 = self.bu_blend_p4(torch.cat([bu_p3_down, td_p4], dim=1))

        bu_p4_down = self.bu_down_p4(bu_p4)
        bu_p5 = self.bu_blend_p5(torch.cat([bu_p4_down, td_p5], dim=1))

        return bu_p3, bu_p4, bu_p5


class FPNPAN4(nn.Module):
    """
    4-Level Feature Pyramid Network with Path Aggregation (P2, P3, P4, P5).

    Extends FPN+PAN with a high-resolution Stride-4 (P2) level (104×104 resolution at 416px).
    Designed specifically for tiny drone objects (<20px) where stride 8 (P3) loses spatial detail.

    Architecture:
        Top-down  (FPN): P5 → P4 → P3 → P2   (semantic context enrichment)
        Bottom-up (PAN): P2 → P3 → P4 → P5   (high-resolution spatial propagation)

    Args:
        in_channels_list : Channel counts for [P2, P3, P4, P5] from the backbone.
        out_channels     : Uniform output channel count for all 4 pyramid levels.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
    ) -> None:
        super().__init__()
        c2_in, c3_in, c4_in, c5_in = in_channels_list

        # ── Top-down (FPN) lateral projections ─────────────────────
        self.lat_c5 = ConvBNAct(c5_in, out_channels, kernel_size=1, padding=0)
        self.lat_c4 = ConvBNAct(c4_in, out_channels, kernel_size=1, padding=0)
        self.lat_c3 = ConvBNAct(c3_in, out_channels, kernel_size=1, padding=0)
        self.lat_c2 = ConvBNAct(c2_in, out_channels, kernel_size=1, padding=0)

        # Top-down fusion CSP blocks
        self.td_blend_p4 = CSPBlock(out_channels, out_channels, n_blocks=1)
        self.td_blend_p3 = CSPBlock(out_channels, out_channels, n_blocks=1)
        self.td_blend_p2 = CSPBlock(out_channels, out_channels, n_blocks=1)

        # ── Bottom-up (PAN) downsampling convolutions ──────────────
        self.bu_down_p2 = ConvBNAct(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bu_down_p3 = ConvBNAct(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bu_down_p4 = ConvBNAct(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

        # Bottom-up fusion CSP blocks
        self.bu_blend_p3 = CSPBlock(out_channels * 2, out_channels, n_blocks=1)
        self.bu_blend_p4 = CSPBlock(out_channels * 2, out_channels, n_blocks=1)
        self.bu_blend_p5 = CSPBlock(out_channels * 2, out_channels, n_blocks=1)

    def forward(
        self,
        features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c2, c3, c4, c5 = features

        # ── 1. Top-down FPN ─────────────────────────────────────────
        td_p5 = self.lat_c5(c5)

        td_p5_up = F.interpolate(td_p5, size=c4.shape[2:], mode="nearest")
        td_p4 = self.td_blend_p4(self.lat_c4(c4) + td_p5_up)

        td_p4_up = F.interpolate(td_p4, size=c3.shape[2:], mode="nearest")
        td_p3 = self.td_blend_p3(self.lat_c3(c3) + td_p4_up)

        td_p3_up = F.interpolate(td_p3, size=c2.shape[2:], mode="nearest")
        td_p2 = self.td_blend_p2(self.lat_c2(c2) + td_p3_up)

        # ── 2. Bottom-up PAN ────────────────────────────────────────
        bu_p2 = td_p2

        bu_p2_down = self.bu_down_p2(bu_p2)
        bu_p3 = self.bu_blend_p3(torch.cat([bu_p2_down, td_p3], dim=1))

        bu_p3_down = self.bu_down_p3(bu_p3)
        bu_p4 = self.bu_blend_p4(torch.cat([bu_p3_down, td_p4], dim=1))

        bu_p4_down = self.bu_down_p4(bu_p4)
        bu_p5 = self.bu_blend_p5(torch.cat([bu_p4_down, td_p5], dim=1))

        return bu_p2, bu_p3, bu_p4, bu_p5

