"""Run the official baseline train.py with OUR split rules (used for E0 and P1.6).

The official script trains on all training terrains and validates on `val` every epoch.
We need: train WITHOUT the dev terrains, and no look at `val` during training.
So we replace its `make_loaders` function: train = train split minus dev terrains,
"val" = the dev terrains (train split, no augmentation). The official code is not changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from raynet.splits import DEV_FILE, load_terrains


def filter_samples(ds: Any, terrains: set[str], keep: bool) -> int:
    """Keep (keep=True) or drop (keep=False) the samples of the given terrains; return count."""
    ds.samples = [
        (i, b) for (i, b) in ds.samples if (ds.rows[i]["terrain_id"] in terrains) == keep
    ]
    return len(ds.samples)


def terrains_of(ds: Any) -> set[str]:
    """Set of terrain ids that appear in the samples of a dataset."""
    return {ds.rows[i]["terrain_id"] for i, _ in ds.samples}


def make_loaders_factory(dev: set[str], info: dict):
    """Return a replacement for the official make_loaders(ld, args)."""

    def make_loaders(ld, args):
        """Train = train minus dev terrains; monitor = dev terrains (no val split)."""
        from torch.utils.data import DataLoader

        common = {"root": args.data_root, "band": args.band, "return_mask": True}
        train_ds = ld.LunarRadioMapDataset(split="train", augment=not args.no_augment, **common)
        dev_ds = ld.LunarRadioMapDataset(split="train", augment=False, **common)
        n_train = filter_samples(train_ds, dev, keep=False)
        n_dev = filter_samples(dev_ds, dev, keep=True)
        overlap = terrains_of(train_ds) & terrains_of(dev_ds)
        if overlap or n_dev == 0:
            raise SystemExit(f"bad split: {len(overlap)} shared terrains, {n_dev} dev samples")
        info.update(train_samples=n_train, dev_samples=n_dev)
        print(f"[run_official] samples: {n_train} train (dev removed) / {n_dev} dev "
              f"(shown as 'val' in the logs; the real val split is NOT used)")
        dl = {"batch_size": args.batch_size, "num_workers": args.num_workers,
              "pin_memory": True, "persistent_workers": args.num_workers > 0,
              "drop_last": False}
        return (DataLoader(train_ds, shuffle=True, **dl),
                DataLoader(dev_ds, shuffle=False, **dl), train_ds)

    return make_loaders


def git_commit(path: Path) -> str:
    """Short git commit hash of a folder, or 'unknown'."""
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def finish_official_args(a: argparse.Namespace) -> argparse.Namespace:
    """Same defaults as the official `if __name__ == "__main__"` block."""
    if a.run_name is None:
        a.run_name = f"radiownet_{a.band}_{a.loss}_{a.phase}"
    if a.init_from is None and a.phase == "secondU":
        a.init_from = "auto"
    if a.init_from in ("none", "scratch", ""):
        a.init_from = None
    if a.init_from == "auto":
        raise SystemExit("--init-from auto prefers the official _best.pt (chosen on val). "
                         "Pass the firstU _last.pt path by hand.")
    return a


def main(argv: list[str] | None = None) -> None:
    """Parse our options, then run the official train.main with the patched loaders."""
    argv = sys.argv[1:] if argv is None else argv
    ours, theirs = (argv[: argv.index("--")], argv[argv.index("--") + 1 :]) if "--" in argv \
        else (argv, [])
    p = argparse.ArgumentParser(description="Official train.py with our dev split.")
    p.add_argument("--official", type=Path, required=True, help="clone of the official repo")
    p.add_argument("--dev-file", type=Path, default=DEV_FILE)
    args = p.parse_args(ours)

    dev = load_terrains(args.dev_file)
    sys.path.insert(0, str(args.official.resolve()))
    import train as official  # the official train.py

    info: dict = {
        "dev_file": str(args.dev_file), "dev_terrains": len(dev),
        "dev_file_sha1": hashlib.sha1(args.dev_file.read_bytes()).hexdigest()[:12],
        "official_commit": git_commit(args.official), "our_commit": git_commit(Path.cwd()),
        "official_args": theirs,
    }
    official.make_loaders = make_loaders_factory(dev, info)
    a = finish_official_args(official.build_parser().parse_args(theirs))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    info_path = Path(a.out) / f"{a.run_name}_wrapper.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    official.main(a)
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
