"""
src/models/detector.py

VanillaDroneDetector — End-to-End Drone Detection Model.

Assembles Backbone + Neck + Heads into a single trainable module.
The architecture variant (neck type, CBAM) is driven entirely by the
config dictionary, making it trivial to switch between Model-A/B/C
by swapping the YAML config.

Pipeline:
    Input Image (B, 3, H, W)
        ↓
    ContextEnhancedBackbone  → P3, P4, P5
        ↓
    FPN / FPNPAN Neck        → fused P3, P4, P5
        ↓
    DecoupledHead × 3        → per-scale {cls, reg, obj} predictions
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn

from src.models.backbone import ContextEnhancedBackbone
from src.models.neck import FPN, FPNPAN, FPNPAN4
from src.models.head import DecoupledHead


# Neck registry — maps config string to class
_NECK_REGISTRY: Dict[str, type] = {
    "fpn": FPN,
    "fpnpan": FPNPAN,
    "fpnpan4": FPNPAN4,
}

# Default strides for standard 3-level pyramid
_DEFAULT_STRIDES: List[int] = [8, 16, 32]


class VanillaDroneDetector(nn.Module):
    """
    Vanilla from-scratch Drone Detector.

    All weights are randomly initialized — no pretrained components.

    Args:
        cfg: OmegaConf / dict with model configuration. Expected keys:
             - num_classes   (int)  : Number of object classes. Default: 1.
             - base_channels (int)  : Backbone base channel width. Default: 32.
             - neck_type     (str)  : "fpn", "fpnpan", or "fpnpan4".
             - use_cbam      (bool) : Whether to use CBAM in backbone.

    Example::
        cfg = {"num_classes": 1, "base_channels": 32, "neck_type": "fpnpan4", "use_cbam": True}
        model = VanillaDroneDetector(cfg)
        outputs = model(torch.randn(2, 3, 416, 416))
    """

    _FPN_OUT_CHANNELS: int = 256

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()

        num_classes: int = cfg.get("num_classes", 1)
        base_channels: int = cfg.get("base_channels", 32)
        neck_type: str = cfg.get("neck_type", "fpn")
        use_cbam: bool = cfg.get("use_cbam", False)

        if neck_type not in _NECK_REGISTRY:
            raise ValueError(
                f"Unknown neck_type '{neck_type}'. Choose from: {list(_NECK_REGISTRY.keys())}"
            )

        self.is_4_level = (neck_type == "fpnpan4")
        self.strides: List[int] = [4, 8, 16, 32] if self.is_4_level else [8, 16, 32]

        # ── Backbone ───────────────────────────────────────────────
        self.backbone = ContextEnhancedBackbone(
            in_channels=3,
            base_channels=base_channels,
            use_cbam=use_cbam,
            return_c2=self.is_4_level,
        )

        if self.is_4_level:
            # Backbone output channels: [base*2 (C2), base*4 (C3), base*8 (C4), base*16 (C5)]
            backbone_channels = [
                base_channels * 2,
                base_channels * 4,
                base_channels * 8,
                base_channels * 16,
            ]
        else:
            # Backbone output channels: [base*4 (C3), base*8 (C4), base*16 (C5)]
            backbone_channels = [
                base_channels * 4,
                base_channels * 8,
                base_channels * 16,
            ]

        # ── Neck ───────────────────────────────────────────────────
        NeckClass = _NECK_REGISTRY[neck_type]
        self.neck = NeckClass(backbone_channels, self._FPN_OUT_CHANNELS)

        # ── Detection Heads (one per pyramid level) ────────────────
        self.heads = nn.ModuleList([
            DecoupledHead(
                num_classes=num_classes,
                in_channels=self._FPN_OUT_CHANNELS,
                hidden_channels=self._FPN_OUT_CHANNELS,
            )
            for _ in self.strides
        ])

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Full forward pass through the detection pipeline.

        Args:
            x: Input image batch of shape (B, 3, H, W).

        Returns:
            List of prediction dicts, one per pyramid scale:
            [
                {"stride": s, "cls": (B, C, H/s, W/s), "reg": (B, 4, H/s, W/s), "obj": (B, 1, H/s, W/s)},
                ...
            ]
        """
        # Step 1: Multi-scale feature extraction
        backbone_feats = self.backbone(x)

        # Step 2: Feature pyramid fusion
        pyramid_feats = self.neck(backbone_feats)

        # Step 3: Per-scale prediction
        outputs = []
        for feat, head in zip(pyramid_feats, self.heads):
            cls_out, reg_out, obj_out = head(feat)
            outputs.append({
                "cls": cls_out,
                "reg": reg_out,
                "obj": obj_out,
            })

        return outputs

