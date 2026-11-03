"""Physics features: free-space loss, knife-edge term, polar view, horizon scan (docs/02_PHYSICS.md).

Clean version of the verified prototype `docs/reference/physics_prototype.py` (same maths).
Conventions: (row, col) = (y, x); pixel centres at integer coordinates; metres, Hz, dB.
Everything here is ONNX-exportable: no torch.cummax and no value-dependent Python branches.
Note: the dataset ray tracer has diffraction DISABLED, so the knife-edge term J is a
"how deep in shadow" feature, not the true loss.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

C_M_S = 299_792_458.0  # speed of light
TX_H_M, RX_H_M = 3.0, 1.0  # antenna heights above the terrain (dataset setup)
NEG = -1.0e4  # "minus infinity" that is safe in ONNX
BAND_HZ = {"415": 415.0e6, "58": 5.775e9}  # same as metadata.json "bands_hz"
FEATURES = ("fspl_db", "los", "shadow_rad", "nu", "j_db", "prior_db")


def fspl_db(d_m: torch.Tensor, f_hz: torch.Tensor) -> torch.Tensor:
    """Free-space pathloss in dB; distance clamped to at least 1 m."""
    return 20 * torch.log10(torch.clamp(d_m, min=1.0)) + 20 * torch.log10(f_hz) - 147.55


def knife_edge_db(nu: torch.Tensor) -> torch.Tensor:
    """Single knife-edge loss J(nu) in dB (ITU-R P.526 approximation); 0 for nu <= -0.78."""
    j = 6.9 + 20 * torch.log10(torch.sqrt((nu - 0.1) ** 2 + 1) + nu - 0.1)
    return torch.where(nu > -0.78, j, torch.zeros_like(nu))


def fresnel_nu(
    h_m: torch.Tensor, d1_m: torch.Tensor, d2_m: torch.Tensor, lam_m: torch.Tensor
) -> torch.Tensor:
    """Fresnel-Kirchhoff parameter nu = h * sqrt(2 (d1 + d2) / (lambda d1 d2))."""
    return h_m * torch.sqrt(2 * (d1_m + d2_m) / (lam_m * d1_m * d2_m))


def scan_max_exclusive(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exclusive running max along the last dim and its index (float); ONNX-safe.

    m[..., k] = max(a[..., :k]) (NEG for k = 0). Hillis-Steele scan: log2(R) "shift and
    compare" steps. The loop length depends only on the shape, so it unrolls in ONNX.
    """
    r = a.shape[-1]
    idx = torch.arange(r, device=a.device, dtype=torch.float32).expand_as(a)
    m = torch.cat([torch.full_like(a[..., :1], NEG), a[..., :-1]], dim=-1)
    i = torch.cat([torch.zeros_like(idx[..., :1]), idx[..., :-1]], dim=-1)
    s = 1
    while s < r:
        m_sh = torch.cat([torch.full_like(m[..., :s], NEG), m[..., :-s]], dim=-1)
        i_sh = torch.cat([torch.zeros_like(i[..., :s]), i[..., :-s]], dim=-1)
        take = m_sh > m
        m = torch.where(take, m_sh, m)
        i = torch.where(take, i_sh, i)
        s *= 2
    return m, i


def scan_max_inclusive(a: torch.Tensor) -> torch.Tensor:
    """Inclusive running max along the last dim (= torch.cummax values), ONNX-safe."""
    return torch.maximum(scan_max_exclusive(a)[0], a)


class PhysicsFeatures(nn.Module):
    """Physics features of 256x256 tiles on a TX-centred polar grid (T rays, R radii).

    forward(height (B,1,S,S) m, tx (B,2) (row, col), freq_hz (B,)) -> (B, 6, S, S) with the
    channels in FEATURES: FSPL dB, LoS 0/1, shadow (rad), nu, J dB, prior = FSPL + J dB.
    """

    def __init__(self, size: int = 256, n_theta: int = 720, dr: float = 1.0,
                 r_max: float = 363.0) -> None:
        """Build the fixed polar and Cartesian grids (buffers)."""
        super().__init__()
        self.size, self.n_theta, self.dr = size, n_theta, dr
        self.n_r = math.ceil(r_max / dr)
        th = 2 * math.pi * torch.arange(n_theta, dtype=torch.float32) / n_theta
        ii, jj = torch.meshgrid(torch.arange(size, dtype=torch.float32),
                                torch.arange(size, dtype=torch.float32), indexing="ij")
        self.register_buffer("r", (torch.arange(self.n_r, dtype=torch.float32) + 1) * dr)
        self.register_buffer("cos", torch.cos(th))
        self.register_buffer("sin", torch.sin(th))
        self.register_buffer("ii", ii)
        self.register_buffer("jj", jj)

    def tx_height(self, height: torch.Tensor, tx: torch.Tensor) -> torch.Tensor:
        """TX antenna height z_tx = H(tx) + 3 m, shape (B, 1, 1)."""
        b, s = height.shape[0], self.size
        g = torch.stack([tx[:, 1] / (s - 1) * 2 - 1, tx[:, 0] / (s - 1) * 2 - 1], dim=-1)
        z = F.grid_sample(height, g.view(b, 1, 1, 2), mode="bilinear", align_corners=True)
        return z.view(b, 1, 1) + TX_H_M

    def polar(self, height: torch.Tensor, tx: torch.Tensor,
              freq_hz: torch.Tensor) -> dict[str, torch.Tensor]:
        """Polar maps (B, T, R): terrain, angles, horizon, LoS, shadow, nu, inside-tile mask."""
        b, s = height.shape[0], self.size
        tr, tc = tx[:, 0].view(b, 1, 1), tx[:, 1].view(b, 1, 1)
        yy = tr + self.sin.view(1, -1, 1) * self.r.view(1, 1, -1)
        xx = tc + self.cos.view(1, -1, 1) * self.r.view(1, 1, -1)
        grid = torch.stack([xx / (s - 1) * 2 - 1, yy / (s - 1) * 2 - 1], dim=-1)  # (x, y)!
        hs = F.grid_sample(height, grid, mode="bilinear", padding_mode="border",
                           align_corners=True)[:, 0]
        inside = ((xx >= 0) & (xx <= s - 1) & (yy >= 0) & (yy <= s - 1)).float()
        z_tx = self.tx_height(height, tx)
        r = self.r.view(1, 1, -1).expand_as(hs)
        alpha = torch.atan2(hs - z_tx, r)  # terrain elevation angle seen from the TX
        horizon, idx = scan_max_exclusive(alpha)  # highest angle before sample k, and where
        z_rx = hs + RX_H_M
        beta = torch.atan2(z_rx - z_tx, r)  # receiver elevation angle
        los = (beta >= horizon).float()
        shadow = torch.clamp(horizon - beta, min=0.0)
        # dominant edge = the TX-horizon point -> Fresnel-Kirchhoff parameter nu
        r_edge = (idx + 1.0) * self.dr
        d1 = torch.clamp(r_edge, min=1.0)
        d2 = torch.clamp(r - r_edge, min=1.0)
        h_edge = torch.gather(hs, 2, idx.long())
        z_line = z_tx + (z_rx - z_tx) * (r_edge / r)
        lam = (C_M_S / freq_hz).view(b, 1, 1)
        nu = fresnel_nu(h_edge - z_line, d1, d2, lam)
        nu = torch.where(horizon < -1.0, torch.full_like(nu, -10.0), nu)  # k = 0: no edge
        return {"height": hs, "alpha": alpha, "horizon": horizon, "beta": beta, "los": los,
                "shadow": shadow, "nu": nu, "inside": inside, "z_tx": z_tx}

    def to_cartesian(self, polar: torch.Tensor, tx: torch.Tensor) -> torch.Tensor:
        """Sample polar maps (B, C, T, R) at every pixel's (r, theta) -> (B, C, S, S)."""
        b, s = polar.shape[0], self.size
        polar = torch.cat([polar, polar[:, :, :1, :]], dim=2)  # wrap theta (2 pi = 0)
        dy = self.ii.view(1, s, s) - tx[:, 0].view(b, 1, 1)
        dx = self.jj.view(1, s, s) - tx[:, 1].view(b, 1, 1)
        rr = torch.sqrt(dx * dx + dy * dy)
        tt = torch.atan2(dy, dx)
        tt = torch.where(tt < 0, tt + 2 * math.pi, tt)
        kr = torch.clamp(rr / self.dr - 1.0, 0.0, self.n_r - 1.0)  # radius index
        kt = tt / (2 * math.pi) * self.n_theta  # angle index in [0, T]
        g = torch.stack([kr / (self.n_r - 1) * 2 - 1, kt / self.n_theta * 2 - 1], dim=-1)
        return F.grid_sample(polar, g, mode="bilinear", padding_mode="border", align_corners=True)

    def distance_3d(self, height: torch.Tensor, tx: torch.Tensor,
                    z_tx: torch.Tensor) -> torch.Tensor:
        """3D TX -> RX distance in metres for every pixel, shape (B, 1, S, S)."""
        b, s = height.shape[0], self.size
        dy = self.ii.view(1, s, s) - tx[:, 0].view(b, 1, 1)
        dx = self.jj.view(1, s, s) - tx[:, 1].view(b, 1, 1)
        rr = torch.sqrt(dx * dx + dy * dy).unsqueeze(1)
        return torch.sqrt(rr**2 + (height + RX_H_M - z_tx.view(b, 1, 1, 1)) ** 2)

    def forward(self, height: torch.Tensor, tx: torch.Tensor,
                freq_hz: torch.Tensor) -> torch.Tensor:
        """The 6 Cartesian feature maps listed in FEATURES (float32)."""
        p = self.polar(height, tx, freq_hz)
        cart = self.to_cartesian(torch.stack([p["los"], p["shadow"], p["nu"]], dim=1), tx)
        los = (cart[:, 0:1] >= 0.5).float()
        nu = cart[:, 2:3]
        j = knife_edge_db(nu)
        fspl = fspl_db(self.distance_3d(height, tx, p["z_tx"]),
                       freq_hz.view(-1, 1, 1, 1))
        return torch.cat([fspl, los, cart[:, 1:2], nu, j, fspl + j], dim=1)
