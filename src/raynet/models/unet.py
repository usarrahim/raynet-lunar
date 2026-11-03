"""SmallUNet: the shared backbone of E2-E5 (docs/03_MODEL.md)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """2 x (Conv3x3 -> GroupNorm(8) -> SiLU)."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.SiLU(),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.SiLU(),
    )


class Up(nn.Module):
    """Bilinear x2 + Conv3x3, concat with the skip, then a conv block."""

    def __init__(self, c_in: int, c_skip: int, c_out: int) -> None:
        """Layers of one decoder level."""
        super().__init__()
        self.reduce = nn.Conv2d(c_in, c_skip, 3, padding=1)
        self.block = conv_block(2 * c_skip, c_out)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample x to the skip size and fuse."""
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.block(torch.cat([self.reduce(x), skip], dim=1))


class SmallUNet(nn.Module):
    """U-Net with 4 down levels (256 -> 16), channels [w, 2w, 4w, 8w], bottleneck 8w."""

    def __init__(self, in_ch: int, width: int = 32, out_ch: int = 1,
                 zero_head: bool = False) -> None:
        """`zero_head=True` starts with output 0 (used for residual models)."""
        super().__init__()
        c = [width, 2 * width, 4 * width, 8 * width]
        self.enc = nn.ModuleList([conv_block(in_ch, c[0]), conv_block(c[0], c[1]),
                                  conv_block(c[1], c[2]), conv_block(c[2], c[3])])
        self.bottleneck = conv_block(c[3], c[3])
        self.up = nn.ModuleList([Up(c[3], c[3], c[2]), Up(c[2], c[2], c[1]),
                                 Up(c[1], c[1], c[0]), Up(c[0], c[0], c[0])])
        self.head = nn.Conv2d(c[0], out_ch, 1)
        if zero_head:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_ch, H, W) -> (B, out_ch, H, W); H and W must be divisible by 16."""
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = self.bottleneck(F.max_pool2d(x, 2))
        for up, skip in zip(self.up, reversed(skips), strict=True):
            x = up(x, skip)
        return self.head(x)


def count_params(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
