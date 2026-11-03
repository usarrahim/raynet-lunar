"""Masked metrics in dB: pooled RMSE, regions, distance bins, coverage (docs/05_EVALUATION.md).

All inputs are batched tensors of shape (B, 1, H, W) on any device:
pred and gt in dB, mask with 1 = valid (ray-traced) pixel, los with 1 = line of sight.
Only valid pixels count, and sums are divided by the number of valid pixels.
"""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as F

REGIONS = ("los_core", "border", "nlos_core")
BORDER_RADIUS_PX = 3
DIST_EDGES_M = (0.0, 25.0, 50.0, 100.0, 150.0, 200.0, 363.0)
COVER_PL_DB = 120.0  # covered if PL <= 120 dB, that is RSS >= -100 dBm (RSS = 20 - PL)


def disk_kernel(radius: int = BORDER_RADIUS_PX) -> torch.Tensor:
    """(1, 1, 2r+1, 2r+1) kernel with ones inside the Euclidean disk of the given radius."""
    d = torch.arange(-radius, radius + 1, dtype=torch.float32)
    return ((d[:, None] ** 2 + d[None, :] ** 2) <= radius**2).float()[None, None]


def boundary(los: torch.Tensor) -> torch.Tensor:
    """Pixels whose LoS value differs from at least one 4-neighbour inside the grid."""
    los = los.bool()
    b = torch.zeros_like(los)
    dv = los[..., 1:, :] != los[..., :-1, :]
    dh = los[..., :, 1:] != los[..., :, :-1]
    b[..., 1:, :] |= dv
    b[..., :-1, :] |= dv
    b[..., :, 1:] |= dh
    b[..., :, :-1] |= dh
    return b


def region_masks(los: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
    """Disjoint regions inside the valid mask: los_core, border, nlos_core (bool tensors)."""
    los, valid = los.bool(), valid.bool()
    k = disk_kernel().to(los.device)
    near = F.conv2d(boundary(los).float(), k, padding=k.shape[-1] // 2) > 0
    return {
        "los_core": los & ~near & valid,
        "border": near & valid,
        "nlos_core": ~los & ~near & valid,
    }


def distance_map(tx: torch.Tensor, h: int = 256, w: int = 256) -> torch.Tensor:
    """Horizontal distance in metres (1 m/px) from the TX (B, 2) = (row, col) to every pixel."""
    rows = torch.arange(h, device=tx.device, dtype=torch.float32)[None, :, None]
    cols = torch.arange(w, device=tx.device, dtype=torch.float32)[None, None, :]
    r = tx[:, 0, None, None].float()
    c = tx[:, 1, None, None].float()
    return torch.sqrt((rows - r) ** 2 + (cols - c) ** 2)[:, None]


def distance_bin_masks(tx: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
    """Valid pixels split into the distance bins of DIST_EDGES_M (last bin is closed)."""
    dist = distance_map(tx, valid.shape[-2], valid.shape[-1])
    out = {}
    edges = DIST_EDGES_M
    for i, (lo, hi) in enumerate(itertools.pairwise(edges)):
        upper = dist <= hi if i == len(edges) - 2 else dist < hi
        out[f"{lo:.0f}-{hi:.0f}m"] = (dist >= lo) & upper & valid.bool()
    return out


def _per_map(x: torch.Tensor) -> list[float]:
    """Sum over (1, H, W) of each map, as Python floats (float64 for accuracy)."""
    return x.double().sum(dim=(1, 2, 3)).cpu().tolist()


def rmse(sse: float, n: float) -> float:
    """sqrt(sse / n), or NaN when there are no pixels."""
    return math.sqrt(sse / n) if n > 0 else float("nan")


def combined_score(rmse_415: float, rmse_58: float) -> float:
    """Challenge score: 0.5 * RMSE(415 MHz) + 0.5 * RMSE(5.8 GHz)."""
    return 0.5 * rmse_415 + 0.5 * rmse_58


class MetricAccumulator:
    """Collects per-map sums over batches; gives a summary and one row per map."""

    def __init__(self, cover_thresholds_db: tuple[float, ...] = (COVER_PL_DB,)) -> None:
        """Start empty. Coverage is computed for every threshold (PL in dB)."""
        self.cover_thresholds = cover_thresholds_db
        self.rows: list[dict] = []
        self.dist_sums: dict[str, list[float]] = {}

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor,
        los: torch.Tensor | None = None,
        tx: torch.Tensor | None = None,
        sample_ids: list[str] | None = None,
        terrain_ids: list[str] | None = None,
    ) -> None:
        """Add one batch. Regions need `los`; distance bins need `tx`."""
        pred, gt = pred.float(), gt.float()
        valid = mask.bool()
        err = pred - gt
        se = err**2
        b = pred.shape[0]
        cols: dict[str, list[float]] = {
            "n_valid": _per_map(valid),
            "sse_all": _per_map(se * valid),
            "sum_err_all": _per_map(err * valid),
            "sse_allpx": _per_map(se),
            "n_allpx": [float(se[0].numel())] * b,
        }
        if los is not None:
            for name, m in region_masks(los, valid).items():
                cols[f"n_{name}"] = _per_map(m)
                cols[f"sse_{name}"] = _per_map(se * m)
                cols[f"sum_err_{name}"] = _per_map(err * m)
        for thr in self.cover_thresholds:
            p_nc, g_nc = pred > thr, gt > thr  # "not covered" is the positive class
            cols[f"tp_{thr:g}"] = _per_map(p_nc & g_nc & valid)
            cols[f"tn_{thr:g}"] = _per_map(~p_nc & ~g_nc & valid)
            cols[f"fp_{thr:g}"] = _per_map(p_nc & ~g_nc & valid)
            cols[f"fn_{thr:g}"] = _per_map(~p_nc & g_nc & valid)
        if tx is not None:
            for name, m in distance_bin_masks(tx, valid).items():
                s = self.dist_sums.setdefault(name, [0.0, 0.0, 0.0])
                s[0] += float(m.double().sum())
                s[1] += float((se * m).double().sum())
                s[2] += float((err * m).double().sum())
        for i in range(b):
            row = {
                "sample_id": sample_ids[i] if sample_ids else str(len(self.rows)),
                "terrain_id": terrain_ids[i] if terrain_ids else "",
            }
            row.update({k: v[i] for k, v in cols.items()})
            row["rmse_all"] = rmse(row["sse_all"], row["n_valid"])
            for name in REGIONS:
                if f"n_{name}" in row:
                    row[f"rmse_{name}"] = rmse(row[f"sse_{name}"], row[f"n_{name}"])
            thr = self.cover_thresholds[0]
            right = row[f"tp_{thr:g}"] + row[f"tn_{thr:g}"]
            row["coverage_acc"] = right / row["n_valid"] if row["n_valid"] else float("nan")
            self.rows.append(row)

    def total(self, key: str) -> float:
        """Sum of one column over all maps."""
        return sum(r[key] for r in self.rows)

    def summary(self) -> dict:
        """Pooled metrics over all maps (the main numbers for val_summary.json)."""
        n = self.total("n_valid")
        out: dict = {
            "n_maps": len(self.rows),
            "n_valid_px": n,
            "rmse_db": rmse(self.total("sse_all"), n),
            "bias_db": self.total("sum_err_all") / n if n else float("nan"),
            "rmse_db_allpx_diagnostic": rmse(self.total("sse_allpx"), self.total("n_allpx")),
            "mean_per_map_rmse_db": sum(r["rmse_all"] for r in self.rows) / len(self.rows)
            if self.rows else float("nan"),
        }
        if self.rows and "n_los_core" in self.rows[0]:
            out["regions"] = {}
            for name in REGIONS:
                nr = self.total(f"n_{name}")
                out["regions"][name] = {
                    "rmse_db": rmse(self.total(f"sse_{name}"), nr),
                    "bias_db": self.total(f"sum_err_{name}") / nr if nr else float("nan"),
                    "pixel_share": nr / n if n else float("nan"),
                    "sse_share": self.total(f"sse_{name}") / self.total("sse_all")
                    if self.total("sse_all") else float("nan"),
                }
        if self.dist_sums:
            out["distance_bins"] = {
                name: {"rmse_db": rmse(s[1], s[0]),
                       "bias_db": s[2] / s[0] if s[0] else float("nan"),
                       "pixel_share": s[0] / n if n else float("nan")}
                for name, s in self.dist_sums.items()
            }
        out["coverage"] = {}
        for thr in self.cover_thresholds:
            tp, tn = self.total(f"tp_{thr:g}"), self.total(f"tn_{thr:g}")
            fp, fn = self.total(f"fp_{thr:g}"), self.total(f"fn_{thr:g}")
            f1_den = 2 * tp + fp + fn
            out["coverage"][f"{thr:g}"] = {
                "accuracy": (tp + tn) / n if n else float("nan"),
                "f1_not_covered": 2 * tp / f1_den if f1_den else float("nan"),
            }
        return out
