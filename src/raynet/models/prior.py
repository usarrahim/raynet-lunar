"""Physics prior models: E1 = FSPL + J, E1b = calibrated linear mix (docs/02_PHYSICS.md §7)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn

from raynet.physics import PhysicsFeatures

# order of the E1b design matrix columns (a0 is the constant term)
E1B_TERMS = ("const", "fspl_db", "j_db", "los", "shadow_deg")


def e1b_design(feats: torch.Tensor) -> torch.Tensor:
    """Design matrix columns (B, 5, H, W) from PhysicsFeatures output (B, 6, H, W)."""
    fspl, los, shadow, _, j = (feats[:, k : k + 1] for k in range(5))
    return torch.cat([torch.ones_like(fspl), fspl, j, los, shadow * (180.0 / math.pi)], dim=1)


def load_e1b_coeffs(path: Path | str) -> dict[str, list[float]]:
    """Read {band: [a0..a4]} from results/e1b_coeffs.json."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {band: [float(v) for v in data[band]["coeffs"]] for band in data if band in ("415", "58")}


class PriorModel(nn.Module):
    """Pathloss in dB from physics only; `coeffs` (5 numbers) turns E1 into E1b."""

    def __init__(self, coeffs: list[float] | None = None, n_theta: int = 720) -> None:
        """E1 when coeffs is None, else E1b with PL = sum(a_k * term_k)."""
        super().__init__()
        self.physics = PhysicsFeatures(n_theta=n_theta)
        self.calibrated = coeffs is not None
        c = torch.tensor(coeffs if coeffs is not None else [0.0] * len(E1B_TERMS))
        self.register_buffer("coeffs", c.float().view(1, -1, 1, 1))

    def forward_with_features(
        self, height: torch.Tensor, tx: torch.Tensor, freq_hz: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(PL dB (B,1,H,W), physics features (B,6,H,W)), so callers can reuse LoS."""
        feats = self.physics(height, tx, freq_hz)
        if self.calibrated:
            pl = (e1b_design(feats) * self.coeffs).sum(dim=1, keepdim=True)
        else:
            pl = feats[:, 5:6]
        return pl, feats

    def forward(self, height: torch.Tensor, tx: torch.Tensor, freq_hz: torch.Tensor) -> torch.Tensor:
        """Pathloss in dB, shape (B, 1, H, W)."""
        return self.forward_with_features(height, tx, freq_hz)[0]
