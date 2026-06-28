"""Custom attention modules for Basin-YOLO."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ("CoordAtt",)


class CoordAtt(nn.Module):
    """Coordinate Attention module."""

    def __init__(self, c1: int, c2: int | None = None, reduction: int = 32):
        super().__init__()
        c2 = c2 or c1
        mip = max(8, c1 // reduction)
        self.conv1 = nn.Conv2d(c1, mip, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, c2, 1, 1, 0)
        self.conv_w = nn.Conv2d(mip, c2, 1, 1, 0)
        self.proj = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        _, _, h, w = x.size()
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_h * a_w
