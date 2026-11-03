"""Fit the E1b prior on the DEV split: PL ≈ a0 + a1·FSPL + a2·J + a3·LoS + a4·shadow_deg.

Least squares over all valid dev pixels, one fit per band (normal equations in float64).
Run: python -m raynet.calibrate --data-root data/LunarRM --out results/e1b_coeffs.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from raynet.data import BANDS, make_datasets
from raynet.models.prior import E1B_TERMS, e1b_design
from raynet.physics import BAND_HZ, PhysicsFeatures
from raynet.run_info import git_commit
from raynet.splits import DEV_FILE


@torch.no_grad()
def fit_e1b(ds: Dataset, band: str, device: str = "cpu", batch_size: int = 8,
            num_workers: int = 0, max_batches: int | None = None) -> dict:
    """Least-squares E1b coefficients for one band; also returns the fit RMSE on the data."""
    pf = PhysicsFeatures().to(device)
    k = len(E1B_TERMS)
    xtx = torch.zeros(k, k, dtype=torch.float64, device=device)
    xty = torch.zeros(k, dtype=torch.float64, device=device)
    yty, n = 0.0, 0.0
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        h, tx = batch["height"].to(device), batch["tx"].to(device)
        y, m = batch["pl"].to(device), batch["mask"].to(device).bool()
        freq = torch.full((h.shape[0],), BAND_HZ[band], device=device)
        x = e1b_design(pf(h, tx, freq))  # (B, 5, H, W)
        xv = x.permute(0, 2, 3, 1)[m[:, 0]].double()  # (n_valid, 5)
        yv = y[:, 0][m[:, 0]].double()
        xtx += xv.T @ xv
        xty += xv.T @ yv
        yty += float(yv @ yv)
        n += yv.numel()
    # lstsq instead of solve: still works if a term is constant (for example no shadow at all)
    xtx, xty = xtx.cpu(), xty.cpu()
    coeffs = torch.linalg.lstsq(xtx, xty[:, None], driver="gelsd").solution[:, 0]
    sse = yty - 2 * float(coeffs @ xty) + float(coeffs @ xtx @ coeffs)
    return {"terms": list(E1B_TERMS), "coeffs": coeffs.cpu().tolist(),
            "fit_rmse_db": math.sqrt(max(sse, 0.0) / n), "n_pixels": n}


def main(argv: list[str] | None = None) -> None:
    """Fit both bands on the dev split and save the coefficients as JSON."""
    p = argparse.ArgumentParser(description="Fit the E1b prior on the dev split.")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results/e1b_coeffs.json"))
    p.add_argument("--bands", nargs="+", default=list(BANDS))
    p.add_argument("--dev-file", type=Path, default=DEV_FILE)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=None, help="for smoke tests")
    args = p.parse_args(argv)

    out = {"fit_split": "dev", "commit": git_commit()}
    for band in args.bands:
        ds = make_datasets(args.data_root, band, args.dev_file, augment=False)["dev"]
        out[band] = fit_e1b(ds, band, args.device, args.batch_size, max_batches=args.max_batches)
        coef = ", ".join(f"{t}={c:.4f}" for t, c in zip(E1B_TERMS, out[band]["coeffs"]))
        print(f"band {band}: {coef}; fit RMSE {out[band]['fit_rmse_db']:.3f} dB")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
