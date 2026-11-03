"""
VERIFIED PROTOTYPE (reference only) - RayNet physics features in PyTorch.

Status (checked on 3 Nov 2026, torch 2.15 CPU, onnxruntime 1.29, onnxruntime-web 1.30.0):
  * scan_max_exclusive == exclusive torch.cummax (values and indices), R = 363.
  * Line-of-sight agreement with brute-force line tracing: 99.29 % - 99.95 % of pixels
    on generated lunar terrains (T = 720 rays, dr = 1.0 m).
  * Exports with torch.onnx.export(dynamo=True, opset_version=18). ONNX Runtime (Python)
    matches PyTorch within 0.0012 dB; onnxruntime-web (WASM, 1 thread, headless Chromium)
    matches Python within 0.0002 dB and runs in about 70 ms after warm-up.
  * torch.cummax does NOT export to ONNX (both exporters fail). That is why the scan exists.

How to use: re-implement this cleanly in src/raynet/physics.py (with type hints, docstrings
and the pytest tests listed in docs/02_PHYSICS.md). Keep the maths and conventions the same.
Conventions: (row, col) = (y, x); pixel centres at integer coordinates; metres, Hz, dB.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

C = 299_792_458.0          # speed of light, m/s
TX_H, RX_H = 3.0, 1.0      # antenna heights above terrain, m (dataset setup)
NEG = -1.0e4               # "minus infinity" that is safe in ONNX


def fspl_db(d_m: torch.Tensor, f_hz: torch.Tensor) -> torch.Tensor:
    """Free-space pathloss in dB. d is clamped to >= 1 m."""
    return 20 * torch.log10(torch.clamp(d_m, min=1.0)) + 20 * torch.log10(f_hz) - 147.55


def knife_edge_db(nu: torch.Tensor) -> torch.Tensor:
    """Single knife-edge loss J(nu) in dB (ITU-R P.526 approximation), 0 for nu <= -0.78."""
    j = 6.9 + 20 * torch.log10(torch.sqrt((nu - 0.1) ** 2 + 1) + nu - 0.1)
    return torch.where(nu > -0.78, j, torch.zeros_like(nu))


def scan_max_exclusive(a: torch.Tensor):
    """Exclusive running max and its index along the last dim (ONNX-safe).

    m[..., k] = max(a[..., :k])  (NEG for k = 0),  i[..., k] = argmax position (float).
    Hillis-Steele scan: log2(R) steps of "shift and compare". The Python loop runs over a
    fixed number of steps (depends on the shape only), so it unrolls in the ONNX graph.
    """
    R = a.shape[-1]
    idx = torch.arange(R, device=a.device, dtype=torch.float32).expand_as(a)
    m = torch.cat([torch.full_like(a[..., :1], NEG), a[..., :-1]], dim=-1)
    i = torch.cat([torch.zeros_like(idx[..., :1]), idx[..., :-1]], dim=-1)
    s = 1
    while s < R:
        m_sh = torch.cat([torch.full_like(m[..., :s], NEG), m[..., :-s]], dim=-1)
        i_sh = torch.cat([torch.zeros_like(i[..., :s]), i[..., :-s]], dim=-1)
        take = m_sh > m
        m = torch.where(take, m_sh, m)
        i = torch.where(take, i_sh, i)
        s *= 2
    return m, i


class PhysicsFeatures(nn.Module):
    """Physics features for one 256x256 tile.

    forward(height (B,1,S,S) metres, tx (B,2) float (row, col), freq_hz (B,))
    returns (B,6,S,S): [fspl_db, los(0/1), shadow_rad, nu, J_db, prior_db = fspl + J]
    """

    def __init__(self, size: int = 256, n_theta: int = 720, dr: float = 1.0, r_max: float = 363.0):
        super().__init__()
        self.size, self.T = size, n_theta
        self.R = int(math.ceil(r_max / dr))
        self.dr = dr
        r = (torch.arange(self.R, dtype=torch.float32) + 1) * dr          # r_k = (k+1) dr
        th = 2 * math.pi * torch.arange(n_theta, dtype=torch.float32) / n_theta
        self.register_buffer("r", r)
        self.register_buffer("cos", torch.cos(th))
        self.register_buffer("sin", torch.sin(th))
        ii, jj = torch.meshgrid(torch.arange(size, dtype=torch.float32),
                                torch.arange(size, dtype=torch.float32), indexing="ij")
        self.register_buffer("ii", ii)
        self.register_buffer("jj", jj)

    def forward(self, height, tx, freq_hz):
        B, S = height.shape[0], self.size
        tr = tx[:, 0].view(B, 1, 1)
        tc = tx[:, 1].view(B, 1, 1)

        # 1) sample the terrain along T rays x R radii (polar grid), shape (B, T, R)
        yy = tr + self.sin.view(1, -1, 1) * self.r.view(1, 1, -1)
        xx = tc + self.cos.view(1, -1, 1) * self.r.view(1, 1, -1)
        grid = torch.stack([xx / (S - 1) * 2 - 1, yy / (S - 1) * 2 - 1], dim=-1)  # (x, y) order!
        Hs = F.grid_sample(height, grid, mode="bilinear", padding_mode="border",
                           align_corners=True)[:, 0]
        g_tx = torch.stack([tc / (S - 1) * 2 - 1, tr / (S - 1) * 2 - 1], dim=-1).view(B, 1, 1, 2)
        z_tx = F.grid_sample(height, g_tx, mode="bilinear", align_corners=True).view(B, 1, 1) + TX_H

        # 2) horizon scan: running max of the terrain elevation angle seen from the TX
        r = self.r.view(1, 1, -1).expand_as(Hs)
        alpha = torch.atan2(Hs - z_tx, r)
        A, idx = scan_max_exclusive(alpha)            # horizon before sample k, and where it is
        z_rx = Hs + RX_H
        beta = torch.atan2(z_rx - z_tx, r)            # elevation angle of the receiver
        los_p = (beta >= A).float()
        shadow_p = torch.clamp(A - beta, min=0.0)     # radians below the horizon

        # 3) single dominant edge = the TX-horizon point -> Fresnel-Kirchhoff parameter nu
        r_edge = (idx + 1.0) * self.dr
        d1 = torch.clamp(r_edge, min=1.0)
        d2 = torch.clamp(r - r_edge, min=1.0)
        H_edge = torch.gather(Hs, 2, idx.long())
        z_line = z_tx + (z_rx - z_tx) * (r_edge / r)
        lam = (C / freq_hz).view(B, 1, 1)
        nu = (H_edge - z_line) * torch.sqrt(2 * (d1 + d2) / (lam * d1 * d2))
        nu = torch.where(A < -1.0, torch.full_like(nu, -10.0), nu)  # k = 0: no obstacle yet

        # 4) back to the Cartesian grid: sample polar maps at each pixel's (r, theta)
        polar = torch.stack([los_p, shadow_p, nu], dim=1)            # (B, 3, T, R)
        polar = torch.cat([polar, polar[:, :, :1, :]], dim=2)        # wrap theta (2*pi = 0)
        dy = self.ii.view(1, S, S) - tr
        dx = self.jj.view(1, S, S) - tc
        rr = torch.sqrt(dx * dx + dy * dy)
        tt = torch.atan2(dy, dx)
        tt = torch.where(tt < 0, tt + 2 * math.pi, tt)
        kr = torch.clamp(rr / self.dr - 1.0, 0.0, self.R - 1.0)      # radius index
        kt = tt / (2 * math.pi) * self.T                             # angle index in [0, T]
        g = torch.stack([kr / (self.R - 1) * 2 - 1, kt / self.T * 2 - 1], dim=-1)
        cart = F.grid_sample(polar, g, mode="bilinear", padding_mode="border", align_corners=True)
        los = (cart[:, 0:1] >= 0.5).float()
        shadow = cart[:, 1:2]
        nu_c = cart[:, 2:3]
        J = knife_edge_db(nu_c)

        # 5) free-space loss on the 3D TX->RX distance, prior = FSPL + J
        z_rx_c = height + RX_H
        d3 = torch.sqrt(rr.unsqueeze(1) ** 2 + (z_rx_c - z_tx.view(B, 1, 1, 1)) ** 2)
        fspl = fspl_db(d3, freq_hz.view(B, 1, 1, 1))
        return torch.cat([fspl, los, shadow, nu_c, J, fspl + J], dim=1)


def los_bruteforce(hm: np.ndarray, tr: int, tc: int, step: float = 0.5, chunk: int = 4096) -> np.ndarray:
    """Slow but simple line-of-sight reference (NumPy) for tests.

    For every pixel, sample the straight TX->pixel line every `step` metres (bilinear terrain)
    and check that the terrain never rises above the line. TX at hm[tr,tc]+3 m, RX at +1 m.
    """
    Hh, Ww = hm.shape
    z_tx = hm[tr, tc] + TX_H
    ii, jj = np.mgrid[0:Hh, 0:Ww]
    ii, jj = ii.ravel(), jj.ravel()
    out = np.ones(ii.size, dtype=np.float32)
    nmax = int(np.ceil(np.hypot(Hh, Ww) / step)) + 2

    def bilinear(y, x):
        x = np.clip(x, 0, Ww - 1); y = np.clip(y, 0, Hh - 1)
        x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
        x1 = np.minimum(x0 + 1, Ww - 1); y1 = np.minimum(y0 + 1, Hh - 1)
        wx = x - x0; wy = y - y0
        return (hm[y0, x0] * (1 - wx) * (1 - wy) + hm[y0, x1] * wx * (1 - wy)
                + hm[y1, x0] * (1 - wx) * wy + hm[y1, x1] * wx * wy)

    for s in range(0, ii.size, chunk):
        i, j = ii[s:s + chunk], jj[s:s + chunk]
        n = np.maximum(np.ceil(np.hypot(i - tr, j - tc) / step).astype(int), 1)
        t = np.arange(1, nmax)[None, :] / n[:, None]
        valid = t < 1.0 - 1e-9                         # points strictly between TX and RX
        Hs = bilinear(tr + t * (i - tr)[:, None], tc + t * (j - tc)[:, None])
        zl = z_tx + t * ((hm[i, j] + RX_H) - z_tx)[:, None]
        out[s:s + chunk] = (~((Hs > zl) & valid).any(axis=1)).astype(np.float32)
    return out.reshape(Hh, Ww)
