"""
src/models/blocks.py

Atomic building blocks for the VanillaDroneDetector architecture.
All modules are built entirely from scratch — no pretrained weights.

Components:
    - ConvBNAct  : Conv → BatchNorm → SiLU activation
    - Bottleneck : Residual 1×1 → 3×3 bottleneck
    - CSPBlock   : Cross Stage Partial block for efficient feature reuse
    - ChannelAttention : CBAM channel attention branch
    - SpatialAttention : CBAM spatial attention branch
    - CBAM       : Full Convolutional Block Attention Module (Woo et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """
    Standard Convolution → Batch Normalization → SiLU Activation block.

    SiLU (Swish) provides smooth, non-monotonic gradients which aids
    convergence when training from scratch compared to ReLU.

    Args:
        in_channels  : Number of input feature channels.
        out_channels : Number of output feature channels.
        kernel_size  : Convolution kernel size. Default: 3.
        stride       : Convolution stride. Default: 1.
        padding      : Zero-padding applied to input. Default: 1.
        groups       : Groups for depthwise separable convolution. Default: 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """
    Residual Bottleneck block.

    A lightweight 1×1 → 3×3 residual unit. The shortcut connection
    is only applied when channel dimensions match.

    Args:
        in_channels  : Number of input channels.
        out_channels : Number of output channels.
        shortcut     : Whether to add the residual shortcut. Default: True.
        expansion    : Hidden channel expansion ratio. Default: 0.5.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        self.cv1 = ConvBNAct(in_channels, hidden, kernel_size=1, padding=0)
        self.cv2 = ConvBNAct(hidden, out_channels, kernel_size=3, padding=1)
        self.add = shortcut and (in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class CSPBlock(nn.Module):
    """
    Cross Stage Partial (CSP) Block.

    Splits the input feature map into two pathways:
      - Primary path  : Processed through N Bottleneck residual blocks.
      - Secondary path: Bypassed directly (identity).
    Both are concatenated and fused with a 1×1 convolution.

    CSP reduces gradient redundancy and lowers computational cost
    compared to standard residual networks.

    Args:
        in_channels  : Number of input channels.
        out_channels : Number of output channels.
        n_blocks     : Number of stacked Bottleneck blocks. Default: 1.
        shortcut     : Shortcut flag passed to each Bottleneck. Default: True.
        expansion    : Channel expansion ratio. Default: 0.5.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_blocks: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        self.cv1 = ConvBNAct(in_channels, hidden, kernel_size=1, padding=0)
        self.cv2 = ConvBNAct(in_channels, hidden, kernel_size=1, padding=0)
        self.cv3 = ConvBNAct(2 * hidden, out_channels, kernel_size=1, padding=0)
        self.bottlenecks = nn.Sequential(
            *[Bottleneck(hidden, hidden, shortcut=shortcut, expansion=1.0) for _ in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        primary = self.bottlenecks(self.cv1(x))
        secondary = self.cv2(x)
        return self.cv3(torch.cat([primary, secondary], dim=1))


# ==============================================================================
# CBAM — Convolutional Block Attention Module (Woo et al., ECCV 2018)
# Fully vanilla: no pretrained parameters.
# ==============================================================================

class ChannelAttention(nn.Module):
    """
    Channel Attention branch of CBAM.

    Uses both average and max pooling to capture global context,
    then applies an MLP to produce per-channel attention weights.

    Args:
        channels     : Number of input/output channels.
        reduction    : Channel reduction ratio for the MLP. Default: 16.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        attention = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        return x * attention.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """
    Spatial Attention branch of CBAM.

    Aggregates channel-wise statistics (avg and max) then applies
    a 7×7 convolution to produce a spatial attention map.

    Args:
        kernel_size: Convolution kernel size. Default: 7.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True).values
        pooled = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(pooled))
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al., ECCV 2018).

    Sequentially applies Channel Attention then Spatial Attention.
    Inserted after CSP blocks in Model-C to improve feature selectivity
    for small, low-contrast drone objects.

    Args:
        channels   : Number of feature channels.
        reduction  : Channel reduction ratio. Default: 16.
        kernel_size: Spatial attention kernel size. Default: 7.
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
