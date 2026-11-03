"""Evaluate one model on dev or val with the same metrics for every model (docs/05_EVALUATION.md).

Writes <split>_per_map.csv (one row per map) and <split>_summary.json.
Regions (LoS / border / NLOS) always use the physics LoS on the fine grid (T = 720, dr = 1 m).

Examples:
  python -m raynet.evaluate --model e1  --band 58 --split dev --data-root data/LunarRM
  python -m raynet.evaluate --model e1b --band 58 --split val --data-root data/LunarRM
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from raynet.data import BANDS, make_datasets
from raynet.metrics import MetricAccumulator
from raynet.models.prior import PriorModel, load_e1b_coeffs
from raynet.physics import BAND_HZ, PhysicsFeatures
from raynet.run_info import device_name, git_commit
from raynet.splits import DEV_FILE

COVER_THRESHOLDS_DB = (120.0, 90.0, 100.0, 110.0, 130.0)  # first one is the main threshold

# predict(height, tx, freq_hz) -> (pred_db (B,1,H,W), los (B,1,H,W) or None)
Predictor = Callable[[torch.Tensor, torch.Tensor, torch.Tensor],
                     tuple[torch.Tensor, torch.Tensor | None]]


def prior_predictor(model: PriorModel) -> Predictor:
    """Predictor for E1 / E1b that also returns the LoS it already computed."""
    def predict(h: torch.Tensor, tx: torch.Tensor, f: torch.Tensor):
        pl, feats = model.forward_with_features(h, tx, f)
        return pl, feats[:, 1:2]
    return predict


@torch.no_grad()
def evaluate(predict: Predictor, ds: Dataset, band: str, device: str = "cpu",
             batch_size: int = 8, num_workers: int = 0,
             max_batches: int | None = None) -> MetricAccumulator:
    """Run the predictor over a dataset and collect all metrics."""
    physics = PhysicsFeatures().to(device)  # LoS for regions when the model gives none
    acc = MetricAccumulator(COVER_THRESHOLDS_DB)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        h, tx = batch["height"].to(device), batch["tx"].to(device)
        freq = torch.full((h.shape[0],), BAND_HZ[band], device=device)
        pred, los = predict(h, tx, freq)
        if los is None:
            los = physics(h, tx, freq)[:, 1:2]
        acc.update(pred, batch["pl"].to(device), batch["mask"].to(device), los=los, tx=tx,
                   sample_ids=list(batch["sample_id"]), terrain_ids=list(batch["terrain_id"]))
    return acc


def write_outputs(acc: MetricAccumulator, out_dir: Path, split: str, info: dict) -> dict:
    """Write <split>_per_map.csv and <split>_summary.json; return the summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = acc.rows
    with open(out_dir / f"{split}_per_map.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {**info, **acc.summary()}
    (out_dir / f"{split}_summary.json").write_text(json.dumps(summary, indent=2),
                                                   encoding="utf-8")
    return summary


def build_predictor(args: argparse.Namespace, device: str) -> tuple[Predictor, str]:
    """Model from the command line arguments; returns (predictor, model name)."""
    if args.model == "e1":
        return prior_predictor(PriorModel().to(device).eval()), "E1"
    if args.model == "e1b":
        coeffs = load_e1b_coeffs(args.coeffs)[args.band]
        return prior_predictor(PriorModel(coeffs).to(device).eval()), "E1b"
    raise SystemExit(f"unknown model {args.model!r}")


def main(argv: list[str] | None = None) -> dict:
    """Command line entry point; returns the summary dict."""
    p = argparse.ArgumentParser(description="Evaluate a model with the project metrics.")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--model", choices=["e1", "e1b"], required=True)
    p.add_argument("--coeffs", type=Path, default=Path("results/e1b_coeffs.json"))
    p.add_argument("--band", choices=list(BANDS), required=True)
    p.add_argument("--split", choices=["dev", "val"], required=True)
    p.add_argument("--dev-file", type=Path, default=DEV_FILE)
    p.add_argument("--out", type=Path, default=None, help="default results/eval/<model>_<band>")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=None, help="for smoke tests")
    args = p.parse_args(argv)

    predict, name = build_predictor(args, args.device)
    ds = make_datasets(args.data_root, args.band, args.dev_file, augment=False)[args.split]
    t0 = time.time()
    acc = evaluate(predict, ds, args.band, args.device, args.batch_size, args.num_workers,
                   args.max_batches)
    seconds = time.time() - t0
    info = {"model": name, "band": args.band, "split": args.split, "commit": git_commit(),
            "device": device_name(args.device), "eval_seconds": round(seconds, 1),
            "smoke_test": args.max_batches is not None}
    out = args.out or Path("results/eval") / f"{name}_{args.band}"
    s = write_outputs(acc, out, args.split, info)
    reg = s.get("regions", {})
    print(f"{name} {args.band} {args.split}: RMSE {s['rmse_db']:.3f} dB, bias {s['bias_db']:.3f}"
          f" | los_core {reg['los_core']['rmse_db']:.2f}, border {reg['border']['rmse_db']:.2f},"
          f" nlos_core {reg['nlos_core']['rmse_db']:.2f} | {s['n_maps']} maps, {seconds:.0f} s"
          f" -> {out}")
    return s


if __name__ == "__main__":
    main()
