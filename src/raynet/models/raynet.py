"""RayNet polar branch: TX-centred polar view, ray encoder and learned horizon scan (docs/03_MODEL.md E4).

Tensor layout on the polar grid: (B, C, T, R). The theta axis is "height", the r axis is "width".
The scan runs along r (from the TX outward). In training it uses torch.cummax (fast);
for ONNX export `export_mode = True` switches to the ONNX-safe `scan_max_inclusive`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from raynet.physics import PhysicsFeatures, scan_max_inclusive

POLAR_CHANNELS = ("rel_height", "alpha", "horizon", "los", "shadow", "nu", "inside")


class RayBlock(nn.Module):
    """Dilated conv along each ray, circular conv across rays, GroupNorm, SiLU, residual."""

    def __init__(self, c: int, dilation: int) -> None:
        """Layers of one block."""
        super().__init__()
        self.along = nn.Conv2d(c, c, (1, 5), padding=(0, 2 * dilation), dilation=(1, dilation))
        self.across = nn.Conv2d(c, c, (3, 1))
        self.norm = nn.GroupNorm(8, c)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T, R) -> (B, C, T, R)."""
        y = self.along(x)
        y = F.pad(y, (0, 0, 1, 1), mode="circular")  # theta wraps: ray T-1 is next to ray 0
        return x + self.act(self.norm(self.across(y)))


class RayScan(nn.Module):
    """Learned horizon: running max of learned features along each ray, mixed back in."""

    def __init__(self, c: int, g: int = 16, use_scan: bool = True) -> None:
        """`use_scan=False` replaces the scan by the identity (ablation E5a, same parameters)."""
        super().__init__()
        self.gate = nn.Conv2d(c, g, 1)
        self.mix = nn.Conv2d(c + 2 * g, c, 1)
        self.use_scan = use_scan
        self.export_mode = False

    def scan(self, g: torch.Tensor) -> torch.Tensor:
        """Inclusive running max along r (dim -1)."""
        if not self.use_scan:
            return g
        if self.export_mode:
            return scan_max_inclusive(g)
        return torch.cummax(g, dim=-1).values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """out = F + Conv1x1(concat[F, S, g - S])."""
        g = self.gate(x)
        s = self.scan(g)
        return x + self.mix(torch.cat([x, s, g - s], dim=1))


class PolarBranch(nn.Module):
    """Height + TX -> 32 learned ray features, mapped back to the Cartesian grid."""

    def __init__(self, n_theta: int = 360, n_r: int = 256, channels: int = 32,
                 use_scan: bool = True) -> None:
        """Polar grid T x R (default 360 x 256, dr = 363 / 256 m) and the ray encoder."""
        super().__init__()
        self.physics = PhysicsFeatures(n_theta=n_theta, n_r=n_r)
        c = channels
        self.stem = nn.Conv2d(len(POLAR_CHANNELS), c, (1, 7), padding=(0, 3))
        self.blocks = nn.ModuleList([RayBlock(c, 1), RayScan(c, use_scan=use_scan),
                                     RayBlock(c, 2), RayScan(c, use_scan=use_scan),
                                     RayBlock(c, 4)])
        self.out_channels = c

    def set_export_mode(self, on: bool) -> None:
        """Use the ONNX-safe scan (True) or torch.cummax (False)."""
        for m in self.blocks:
            if isinstance(m, RayScan):
                m.export_mode = on

    def polar_inputs(self, height: torch.Tensor, tx: torch.Tensor,
                     freq_hz: torch.Tensor) -> torch.Tensor:
        """The 7 scaled polar input channels (B, 7, T, R), float32."""
        p = self.physics.polar(height, tx, freq_hz)
        deg = 180.0 / math.pi
        horizon = torch.clamp(p["horizon"], min=-math.pi / 2)  # k = 0 has no horizon yet
        x = [
            (p["height"] - p["z_tx"]) / 50.0,
            p["alpha"] * deg / 30.0,
            horizon * deg / 30.0,
            p["los"],
            torch.clamp(p["shadow"] * deg / 10.0, 0.0, 3.0),
            torch.clamp(p["nu"], -5.0, 5.0) / 5.0,
            p["inside"],
        ]
        return torch.stack(x, dim=1)

    def forward(self, height: torch.Tensor, tx: torch.Tensor, freq_hz: torch.Tensor) -> torch.Tensor:
        """Cartesian ray features (B, C, S, S)."""
        dev = height.device.type
        with torch.autocast(device_type=dev, enabled=False):
            x = self.polar_inputs(height.float(), tx.float(), freq_hz)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        with torch.autocast(device_type=dev, enabled=False):
            return self.physics.to_cartesian(x.float(), tx.float())
