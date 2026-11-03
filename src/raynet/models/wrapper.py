"""Model wrapper: physics input channels -> backbone -> pathloss in dB (docs/02_PHYSICS.md §8).

Normalization happens here and only here. Ranges come from metadata.json (per band).
Physics features are computed in float32 outside autocast (atan2 and log10 are not safe in fp16).
  E2: y_norm = net(x)                 (direct prediction)
  E3: y_norm = prior_norm(E1b) + net(x)  (residual; zero-init head -> starts as the prior)
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from raynet.models.prior import e1b_design
from raynet.models.unet import SmallUNet
from raynet.physics import BAND_HZ, PhysicsFeatures

CHANNELS = ("height", "tx_onehot", "log_distance", "fspl", "los", "shadow", "nu", "j", "prior")
LOG10_RMAX = math.log10(363.0)
KINDS = ("e2", "e3")


@dataclass
class Ranges:
    """Normalization ranges from metadata.json."""

    hm_min: float
    hm_max: float
    pl_min: float
    pl_max: float

    @classmethod
    def from_metadata(cls, root: Path | str, band: str) -> Ranges:
        """Height range and the per-band pathloss range of the dataset."""
        meta = json.loads((Path(root) / "metadata.json").read_text(encoding="utf-8"))
        hm, pl = meta["heightmap_range_m"], meta["pathloss_range_db_per_band"][band]
        return cls(float(hm[0]), float(hm[1]), float(pl[0]), float(pl[1]))


class PathlossNet(nn.Module):
    """(height m, tx (row, col)) -> pathloss dB for one band."""

    def __init__(self, kind: str, band: str, ranges: Ranges, e1b_coeffs: list[float],
                 width: int = 32) -> None:
        """Build physics, backbone and the fixed buffers (frequency, E1b coefficients)."""
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind, self.band, self.ranges, self.width = kind, band, ranges, width
        self.physics = PhysicsFeatures()
        self.net = SmallUNet(len(CHANNELS), width, zero_head=(kind == "e3"))
        self.register_buffer("freq_hz", torch.tensor([BAND_HZ[band]]))
        self.register_buffer("coeffs", torch.tensor(e1b_coeffs).float().view(1, -1, 1, 1))
        self.e1b_coeffs = list(e1b_coeffs)

    def config(self) -> dict:
        """Everything needed to rebuild this model (saved in checkpoints)."""
        return {"kind": self.kind, "band": self.band, "ranges": asdict(self.ranges),
                "e1b_coeffs": self.e1b_coeffs, "width": self.width}

    @classmethod
    def from_config(cls, cfg: dict) -> PathlossNet:
        """Rebuild from `config()`."""
        return cls(cfg["kind"], cfg["band"], Ranges(**cfg["ranges"]), cfg["e1b_coeffs"],
                   cfg["width"])

    def norm_pl(self, pl_db: torch.Tensor) -> torch.Tensor:
        """dB -> normalized pathloss."""
        return (pl_db - self.ranges.pl_min) / (self.ranges.pl_max - self.ranges.pl_min)

    def denorm_pl(self, y: torch.Tensor) -> torch.Tensor:
        """Normalized pathloss -> dB."""
        return y * (self.ranges.pl_max - self.ranges.pl_min) + self.ranges.pl_min

    def inputs(self, height: torch.Tensor, tx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The 9 input channels (B, 9, H, W) and the E1b prior in dB (B, 1, H, W), in fp32."""
        b, _, h, w = height.shape
        height, tx = height.float(), tx.float()
        freq = self.freq_hz.expand(b)
        feats = self.physics(height, tx, freq)
        fspl, los, shadow, nu, j = (feats[:, k : k + 1] for k in range(5))
        prior = (e1b_design(feats) * self.coeffs).sum(dim=1, keepdim=True)
        r = self.ranges
        rows = torch.arange(h, device=height.device, dtype=torch.float32).view(1, 1, h, 1)
        cols = torch.arange(w, device=height.device, dtype=torch.float32).view(1, 1, 1, w)
        onehot = ((rows - tx[:, 0].view(b, 1, 1, 1)).abs() < 0.5) \
            & ((cols - tx[:, 1].view(b, 1, 1, 1)).abs() < 0.5)
        # FSPL = 20 log10(d) + 20 log10(f) - 147.55, so log10(d) comes back exactly
        log_d = (fspl - 20 * torch.log10(freq.view(b, 1, 1, 1)) + 147.55) / 20
        x = torch.cat([
            (height - r.hm_min) / (r.hm_max - r.hm_min),
            onehot.float(),
            log_d / LOG10_RMAX,
            self.norm_pl(fspl),
            los,
            torch.clamp(torch.rad2deg(shadow) / 10, 0, 3),
            torch.clamp(nu, -5, 5) / 5,
            j / 40,
            self.norm_pl(prior),
        ], dim=1)
        return x, prior

    def forward_norm(self, height: torch.Tensor, tx: torch.Tensor) -> torch.Tensor:
        """Normalized prediction y_norm (B, 1, H, W), float32."""
        with torch.autocast(device_type=height.device.type, enabled=False):
            x, prior = self.inputs(height, tx)
        out = self.net(x).float()
        return self.norm_pl(prior) + out if self.kind == "e3" else out

    def forward(self, height: torch.Tensor, tx: torch.Tensor) -> torch.Tensor:
        """Pathloss in dB (B, 1, H, W)."""
        return self.denorm_pl(self.forward_norm(height, tx))
