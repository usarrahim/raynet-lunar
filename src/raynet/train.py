"""Train one model for a FIXED budget (no early stopping); the last checkpoint is the result.

Run: python -m raynet.train --config configs/e3_residual_58.yaml --data-root data/LunarRM
Smoke test: add --epochs 1 --limit-batches 3 --num-workers 0
The run folder runs/<id>_<band>_s<seed>/ gets config.yaml, git_commit.txt, metrics.csv
(one row per epoch), summary.json, last.pt (every epoch, for --resume auto) and best_dev.pt
(lowest dev RMSE, only for curves; never used for reporting).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from raynet.config import load_config, save_config
from raynet.data import make_datasets
from raynet.models.prior import load_e1b_coeffs
from raynet.models.unet import count_params
from raynet.models.wrapper import PathlossNet, Ranges
from raynet.run_info import device_name, git_commit

METRIC_COLS = ("epoch", "train_loss", "dev_rmse_db", "lr", "time_s", "samples_per_s")


def set_seed(seed: int) -> None:
    """Seed random, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_factor(step: int, warmup_steps: int, total_steps: int, lr: float, lr_min: float) -> float:
    """Linear warm-up to 1, then cosine from 1 down to lr_min / lr (multiplier for the base lr)."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    floor = lr_min / lr
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over valid pixels only (divided by mask.sum())."""
    return ((pred - target) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int,
                seed: int, drop_last: bool = False) -> DataLoader:
    """DataLoader with its own seeded generator (same data order after a resume)."""
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=drop_last, generator=g)


@torch.no_grad()
def dev_rmse(model: PathlossNet, loader: DataLoader, device: torch.device, amp: bool,
             max_batches: int | None = None) -> float:
    """Pooled masked RMSE in dB on the dev split (monitor only)."""
    model.eval()
    sse, n = 0.0, 0.0
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        h, tx = batch["height"].to(device), batch["tx"].to(device)
        gt, m = batch["pl"].to(device), batch["mask"].to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            pred = model(h, tx).float()
        sse += float((((pred - gt) ** 2) * m).double().sum())
        n += float(m.double().sum())
    model.train()
    return math.sqrt(sse / n) if n else float("nan")


def save_atomic(obj: dict, path: Path) -> None:
    """torch.save to a temp file, then rename (a crash never leaves a broken last.pt)."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Command line values replace config values (and are saved in config.yaml)."""
    if args.seed is not None:
        cfg["seed"] = args.seed
    for key in ("epochs", "batch_size", "num_workers"):
        if getattr(args, key) is not None:
            cfg["train"][key] = getattr(args, key)
    if args.width is not None:  # only for the P2.2 speed test
        cfg["model"]["width"] = args.width
    return cfg


def budget(cfg: dict) -> dict:
    """The parts of a config that must not change on resume (num_workers may change)."""
    return {"seed": cfg["seed"], "model": cfg["model"],
            "train": {k: v for k, v in cfg["train"].items() if k != "num_workers"}}


def build_model(cfg: dict, data_root: Path) -> PathlossNet:
    """Model from the config; normalization ranges from metadata.json, E1b from results."""
    band = cfg["band"]
    coeffs = load_e1b_coeffs(cfg["data"]["e1b_coeffs"])[band]
    return PathlossNet(cfg["model"]["kind"], band, Ranges.from_metadata(data_root, band),
                       coeffs, cfg["model"]["width"])


def append_metrics(path: Path, row: dict) -> None:
    """Append one row to metrics.csv (header on the first row)."""
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(METRIC_COLS))
        if new:
            w.writeheader()
        w.writerow(row)


def train(cfg: dict, args: argparse.Namespace) -> dict:
    """Full training loop; returns the summary dict."""
    tc = cfg["train"]
    device = torch.device(args.device)
    amp = bool(tc["amp"]) and device.type == "cuda"
    run_name = args.run_name or f"{cfg['id']}_{cfg['band']}_s{cfg['seed']}"
    run_dir = Path(args.out) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])

    sets = make_datasets(args.data_root, cfg["band"], Path(cfg["data"]["dev_file"]),
                         augment=bool(tc["augment"]))
    model = build_model(cfg, args.data_root).to(device)
    params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    n_batches = len(sets["train"]) // tc["batch_size"]
    steps_per_epoch = min(n_batches, args.limit_batches) if args.limit_batches else n_batches
    total = steps_per_epoch * tc["epochs"]
    warm = steps_per_epoch * tc["warmup_epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, warm, total, tc["lr"], tc["lr_min"]))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch, best_dev, best_epoch = 0, float("inf"), -1
    last = run_dir / "last.pt"
    if (args.resume == "auto" and last.exists()) or args.resume not in (None, "auto"):
        ck = torch.load(last if args.resume == "auto" else args.resume, map_location=device,
                        weights_only=False)
        if ck["model_config"] != model.config() or budget(ck["cfg"]) != budget(cfg):
            raise SystemExit("resume: the checkpoint was made with a different config")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_dev, best_epoch = ck["epoch"] + 1, ck["best_dev"], ck["best_epoch"]
        print(f"resumed from {last} at epoch {start_epoch}")
    else:
        save_config(cfg, run_dir / "config.yaml")
        (run_dir / "git_commit.txt").write_text(git_commit() + "\n", encoding="utf-8")
        (run_dir / "metrics.csv").unlink(missing_ok=True)
    print(f"{run_name}: {params:,} params, {len(sets['train'])} train / {len(sets['dev'])} dev "
          f"maps, {steps_per_epoch} steps/epoch, device {device_name(args.device)}, amp {amp}")

    dev_loader = make_loader(sets["dev"], tc["batch_size"], False, tc["num_workers"], 0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    speeds = []
    t_start = time.time()
    for epoch in range(start_epoch, tc["epochs"]):
        loader = make_loader(sets["train"], tc["batch_size"], True, tc["num_workers"],
                             seed=cfg["seed"] * 100_003 + epoch, drop_last=True)
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        lr_epoch = opt.param_groups[0]["lr"]
        for i, batch in enumerate(loader):
            if i >= steps_per_epoch:
                break
            h, tx = batch["height"].to(device, non_blocking=True), batch["tx"].to(device)
            gt, m = batch["pl"].to(device, non_blocking=True), batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                pred = model.forward_norm(h, tx)
            loss = masked_mse(pred.float(), model.norm_pl(gt), m)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += float(loss.detach()) * h.shape[0]
            seen += h.shape[0]
            if args.log_every and i % args.log_every == 0:
                print(f"  ep{epoch} [{i}/{steps_per_epoch}] loss {float(loss.detach()):.6f}",
                      flush=True)
        train_s = time.time() - t0
        speeds.append(seen / train_s)
        dev = float("nan")
        if (epoch + 1) % cfg["eval"]["dev_every"] == 0 or epoch == tc["epochs"] - 1:
            dev = dev_rmse(model, dev_loader, device, amp, args.dev_max_batches)
        if dev < best_dev:
            best_dev, best_epoch = dev, epoch
        row = {"epoch": epoch, "train_loss": running / max(seen, 1), "dev_rmse_db": dev,
               "lr": lr_epoch, "time_s": round(time.time() - t0, 1),
               "samples_per_s": round(seen / train_s, 1)}
        append_metrics(run_dir / "metrics.csv", row)
        state = {"model": model.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                 "best_dev": best_dev, "best_epoch": best_epoch, "cfg": cfg,
                 "model_config": model.config(), "commit": git_commit()}
        save_atomic(state, last)
        if best_epoch == epoch:
            save_atomic({"model": model.state_dict(), "model_config": model.config(),
                         "epoch": epoch, "dev_rmse_db": dev}, run_dir / "best_dev.pt")
        print(f"epoch {epoch}: train_loss {row['train_loss']:.6f}  dev RMSE {dev:.3f} dB  "
              f"({row['time_s']:.0f} s, {row['samples_per_s']:.0f} maps/s)", flush=True)

    summary = {
        "run": run_name, "id": cfg["id"], "band": cfg["band"], "seed": cfg["seed"],
        "kind": cfg["model"]["kind"], "width": cfg["model"]["width"], "params": params,
        "epochs": tc["epochs"], "steps_per_epoch": steps_per_epoch,
        "limit_batches": args.limit_batches, "best_dev_rmse_db": best_dev,
        "best_dev_epoch": best_epoch, "train_hours_this_session": (time.time() - t_start) / 3600,
        "samples_per_s_median": statistics.median(speeds) if speeds else None,
        "gpu": device_name(args.device),
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated(device) / 1e9
        if device.type == "cuda" else None,
        "commit": git_commit(),
    }
    if (run_dir / "metrics.csv").exists():
        with open(run_dir / "metrics.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        summary["final_dev_rmse_db"] = float(rows[-1]["dev_rmse_db"]) if rows else None
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> dict:
    """Command line entry point."""
    p = argparse.ArgumentParser(description="Train one RayNet experiment.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--width", type=int, default=None, help="backbone width (speed test)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--limit-batches", type=int, default=None, help="for smoke tests")
    p.add_argument("--dev-max-batches", type=int, default=None, help="for smoke tests")
    p.add_argument("--resume", default=None, help="'auto' or a checkpoint path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=200)
    args = p.parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args)
    s = train(cfg, args)
    print(json.dumps(s, indent=2))
    return s


if __name__ == "__main__":
    main()
