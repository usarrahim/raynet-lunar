"""Slow brute-force line-of-sight reference for tests (copied from the verified prototype)."""

import numpy as np

TX_H_M, RX_H_M = 3.0, 1.0


def bilinear(hm: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Bilinear terrain height at (y, x), clamped to the tile."""
    hh, ww = hm.shape
    x = np.clip(x, 0, ww - 1)
    y = np.clip(y, 0, hh - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, ww - 1), np.minimum(y0 + 1, hh - 1)
    wx, wy = x - x0, y - y0
    return (hm[y0, x0] * (1 - wx) * (1 - wy) + hm[y0, x1] * wx * (1 - wy)
            + hm[y1, x0] * (1 - wx) * wy + hm[y1, x1] * wx * wy)


def los_bruteforce(hm: np.ndarray, tr: int, tc: int, step: float = 0.5,
                   chunk: int = 4096) -> np.ndarray:
    """LoS 0/1 per pixel: the terrain never rises above the straight TX -> RX line."""
    hh, ww = hm.shape
    z_tx = hm[tr, tc] + TX_H_M
    ii, jj = np.mgrid[0:hh, 0:ww]
    ii, jj = ii.ravel(), jj.ravel()
    out = np.ones(ii.size, dtype=np.float32)
    nmax = int(np.ceil(np.hypot(hh, ww) / step)) + 2
    for s in range(0, ii.size, chunk):
        i, j = ii[s:s + chunk], jj[s:s + chunk]
        n = np.maximum(np.ceil(np.hypot(i - tr, j - tc) / step).astype(int), 1)
        t = np.arange(1, nmax)[None, :] / n[:, None]
        between = t < 1.0 - 1e-9  # points strictly between TX and RX
        hs = bilinear(hm, tr + t * (i - tr)[:, None], tc + t * (j - tc)[:, None])
        zl = z_tx + t * ((hm[i, j] + RX_H_M) - z_tx)[:, None]
        out[s:s + chunk] = (~((hs > zl) & between).any(axis=1)).astype(np.float32)
    return out.reshape(hh, ww)
