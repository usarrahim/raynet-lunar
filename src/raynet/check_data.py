"""Sanity checks for the LunarRM dataset files (docs/01_DATA.md, checks 1-7).

Run: python -m raynet.check_data --root data/LunarRM
Checks whose files are missing are skipped. Exit code 1 if any check fails.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GRID = 256
BANDS = ("415", "58")
CHUNK = 256  # maps read from disk at once
TX_WINDOW = 2  # half size of the window around the TX (5x5 px)
NEAR_TX_TOL_DB = 1.0  # TX window min may be this much above the map min
NEAR_TX_MIN_SHARE = 0.95  # share of maps that must pass check 5
FILL_58_DB = 145.0  # placeholder value of 5.8 GHz void pixels (metadata.json)


@dataclass
class Report:
    """Collects check lines and remembers if any check failed."""

    lines: list[str] = field(default_factory=list)
    failed: bool = False

    def add(self, status: str, name: str, text: str) -> None:
        """Add one line; status is OK, WARN, FAIL, SKIP or INFO."""
        if status == "FAIL":
            self.failed = True
        self.lines.append(f"[{status:4}] {name}: {text}")


def load_metadata(root: Path) -> dict:
    """Read metadata.json from the data root."""
    with open(root / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


def split_files(root: Path, split: str) -> dict[str, Path]:
    """Return the files of one split that exist on disk, keyed by field name."""
    names = {"index": f"{split}_index.csv", "hm": f"{split}_hm.npy"}
    for b in BANDS:
        names[f"pl{b}"] = f"{split}_pl{b}.npy"
        names[f"mask{b}"] = f"{split}_mask{b}.npy"
    return {k: root / v for k, v in names.items() if (root / v).exists()}


def read_index(path: Path, max_rows: int | None = None) -> list[dict[str, str]]:
    """Read an index CSV as a list of dicts (optionally only the first rows)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:max_rows] if max_rows else rows


def unpack_masks(packed: np.ndarray) -> np.ndarray:
    """Unpack bit-packed masks (n, 8192) uint8 to (n, 256, 256) bool, 1 = valid."""
    return np.unpackbits(packed, axis=1).reshape(-1, GRID, GRID).astype(bool)


def chunks(n: int, size: int = CHUNK) -> list[tuple[int, int]]:
    """Split range(n) into (start, stop) pairs of at most `size` items."""
    return [(s, min(s + size, n)) for s in range(0, n, size)]


def check_lengths(rep: Report, split: str, files: dict[str, Path], rows: list[dict]) -> None:
    """Check 1: file sizes, and index length equals array length for every field."""
    for key, path in files.items():
        rep.add("INFO", f"{split} files", f"{path.name} {path.stat().st_size / 1e6:.1f} MB")
    n = len(rows)
    for key, path in files.items():
        if key in ("index", "hm"):
            continue
        rows_arr = np.load(path, mmap_mode="r").shape[0]
        status = "OK" if rows_arr >= n else "FAIL"
        rep.add(status, f"{split} length", f"{path.name} has {rows_arr} rows, index has {n}")
    if "hm" in files:
        n_hm = np.load(files["hm"], mmap_mode="r").shape[0]
        max_row = max(int(r["hm_row"]) for r in rows)
        status = "OK" if max_row < n_hm else "FAIL"
        rep.add(status, f"{split} hm_row", f"max hm_row {max_row}, heightmaps {n_hm}")


def check_heightmap(rep: Report, split: str, path: Path, hm_range: list[float]) -> None:
    """Check 2a: heightmaps are finite and >= 0; compare with the metadata range."""
    hm = np.load(path, mmap_mode="r")
    lo, hi, bad = np.inf, -np.inf, 0
    for s, e in chunks(hm.shape[0]):
        part = np.asarray(hm[s:e])
        bad += int((~np.isfinite(part)).sum())
        lo, hi = min(lo, float(part.min())), max(hi, float(part.max()))
    status = "OK" if bad == 0 and lo >= 0 else "FAIL"
    rep.add(status, f"{split} height", f"min {lo:.2f} m, max {hi:.2f} m, non-finite {bad}")
    if hi > hm_range[1]:
        rep.add("WARN", f"{split} height", f"max above metadata range {hm_range}")


def band_stats(
    pl_path: Path, mask_path: Path | None, rows: list[dict]
) -> dict[str, np.ndarray | int | float]:
    """Read one band in chunks; return value range, valid fractions and near-TX stats."""
    pl = np.load(pl_path, mmap_mode="r")
    masks = np.load(mask_path, mmap_mode="r") if mask_path else None
    n = len(rows)
    tx = np.array([[int(r["tx_row"]), int(r["tx_col"])] for r in rows])
    lo, hi, nonfinite = np.inf, -np.inf, 0
    valid_frac = np.ones(n)
    tx_min, map_min = np.zeros(n), np.zeros(n)
    invalid, invalid_fill = 0, 0
    for s, e in chunks(n):
        part = np.asarray(pl[s:e]).astype(np.float32)
        nonfinite += int((~np.isfinite(part)).sum())
        lo, hi = min(lo, float(np.nanmin(part))), max(hi, float(np.nanmax(part)))
        valid = unpack_masks(np.asarray(masks[s:e])) if masks is not None else np.isfinite(part)
        valid_frac[s:e] = valid.mean(axis=(1, 2))
        invalid += int((~valid).sum())
        invalid_fill += int((part[~valid] == FILL_58_DB).sum())
        masked = np.where(valid, part, np.inf)
        map_min[s:e] = masked.min(axis=(1, 2))
        for i in range(e - s):
            r, c = tx[s + i]
            win = masked[i, max(r - TX_WINDOW, 0) : r + TX_WINDOW + 1,
                         max(c - TX_WINDOW, 0) : c + TX_WINDOW + 1]
            tx_min[s + i] = win.min()
    return {
        "lo": lo, "hi": hi, "nonfinite": nonfinite, "valid_frac": valid_frac,
        "tx_min": tx_min, "map_min": map_min, "invalid": invalid, "invalid_fill": invalid_fill,
    }


def report_band(rep: Report, split: str, band: str, st: dict, pl_range: list[float]) -> None:
    """Checks 2b-5 for one band, from the numbers of `band_stats`."""
    name = f"{split} pl{band}"
    inside = pl_range[0] <= st["lo"] and st["hi"] <= pl_range[1]
    status = "OK" if st["nonfinite"] == 0 and inside else ("FAIL" if split == "train" else "WARN")
    rep.add(status, f"{name} range", f"min {st['lo']:.2f}, max {st['hi']:.2f} dB "
            f"(metadata {pl_range}), non-finite {st['nonfinite']}")
    vf = st["valid_frac"]
    rep.add("INFO", f"{name} valid", f"mean {vf.mean() * 100:.2f}%, median "
            f"{np.median(vf) * 100:.2f}%, min {vf.min() * 100:.2f}%")
    if band == "58" and st["invalid"] > 0:
        share = st["invalid_fill"] / st["invalid"]
        status = "OK" if share > 0.5 else "WARN"
        rep.add(status, f"{name} voids", f"{share * 100:.2f}% of {st['invalid']} invalid px "
                f"are exactly {FILL_58_DB} dB")
    has_valid = np.isfinite(st["tx_min"])
    near = (st["tx_min"] - st["map_min"] <= NEAR_TX_TOL_DB) & has_valid
    share = near.sum() / max(has_valid.sum(), 1)
    status = "OK" if share >= NEAR_TX_MIN_SHARE else "WARN"
    rep.add(status, f"{name} near TX", f"{share * 100:.2f}% of maps have their min PL within "
            f"{NEAR_TX_TOL_DB} dB inside the 5x5 px TX window; median TX-window min "
            f"{np.median(st['tx_min'][has_valid]):.1f} dB")


def check_terrains(rep: Report, split: str, rows: list[dict]) -> None:
    """Check 6: count terrains with 51 TX and with 1 TX; compare with the group column."""
    per_terrain = Counter(r["terrain_id"] for r in rows)
    sizes = Counter(per_terrain.values())
    text = ", ".join(f"{k} TX: {v} terrains" for k, v in sorted(sizes.items(), reverse=True))
    rep.add("INFO", f"{split} terrains", f"{len(per_terrain)} terrains, {len(rows)} maps; {text}")
    if "group" in rows[0]:
        wrong = sum(
            1 for r in rows
            if (r["group"] == "multi") != (per_terrain[r["terrain_id"]] > 1)
        )
        rep.add("OK" if wrong == 0 else "WARN", f"{split} group", f"{wrong} rows disagree "
                "with the 'group' column")


def hillshade(height: np.ndarray) -> np.ndarray:
    """Simple hillshade image of a heightmap (light from the north-west)."""
    from matplotlib.colors import LightSource

    return LightSource(azdeg=315, altdeg=45).hillshade(height, vert_exag=1.0)


def save_figure(root: Path, split: str, rows: list[dict], out: Path, n: int = 6) -> None:
    """Check 7: save a figure with n random samples (height, PL and mask per band)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = split_files(root, split)
    hm = np.load(files["hm"], mmap_mode="r")
    picks = np.random.default_rng(0).choice(len(rows), size=min(n, len(rows)), replace=False)
    cols = ["height"] + [f"{k}{b}" for b in BANDS for k in ("pl", "mask")]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(2.4 * len(cols), 2.4 * len(picks)))
    axes = np.atleast_2d(axes)
    for i, idx in enumerate(picks):
        row = rows[idx]
        tr, tc = int(row["tx_row"]), int(row["tx_col"])
        for j, key in enumerate(cols):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if key == "height":
                ax.imshow(hillshade(np.asarray(hm[int(row["hm_row"])])), cmap="gray")
                ax.set_ylabel(row["sample_id"], fontsize=8)
            elif key in files and key.startswith("pl"):
                im = ax.imshow(np.asarray(np.load(files[key], mmap_mode="r")[idx]),
                               cmap="viridis_r")
                fig.colorbar(im, ax=ax, fraction=0.046)
            elif key in files:
                packed = np.load(files[key], mmap_mode="r")[idx : idx + 1]
                ax.imshow(unpack_masks(np.asarray(packed))[0], cmap="gray", vmin=0, vmax=1)
            ax.plot(tc, tr, "r+", markersize=8)
            if i == 0:
                ax.set_title(key + (" (dB)" if key.startswith("pl") else ""), fontsize=9)
    fig.suptitle(f"{split}: random samples (red + = TX)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)


def check_split(rep: Report, root: Path, split: str, meta: dict, max_maps: int | None) -> None:
    """Run checks 1-6 for one split (skips missing files)."""
    files = split_files(root, split)
    if "index" not in files:
        rep.add("SKIP", split, "no index file")
        return
    rows = read_index(files["index"], max_maps)
    check_lengths(rep, split, files, rows)
    if "hm" in files:
        check_heightmap(rep, split, files["hm"], meta["heightmap_range_m"])
    for b in BANDS:
        if f"pl{b}" not in files:
            rep.add("SKIP", f"{split} pl{b}", "file missing")
            continue
        band_files = [files[k] for k in (f"pl{b}", f"mask{b}") if k in files]
        if any(np.load(f, mmap_mode="r").shape[0] < len(rows) for f in band_files):
            rep.add("SKIP", f"{split} pl{b}", "array shorter than index (see length check)")
            continue
        st = band_stats(files[f"pl{b}"], files.get(f"mask{b}"), rows)
        report_band(rep, split, b, st, meta["pathloss_range_db_per_band"][b])
    check_terrains(rep, split, rows)


def main(argv: list[str] | None = None) -> int:
    """Command line entry point; returns the exit code."""
    p = argparse.ArgumentParser(description="Sanity checks for the LunarRM data files.")
    p.add_argument("--root", type=Path, required=True, help="folder with metadata.json")
    p.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    p.add_argument("--max-maps", type=int, default=None, help="only the first N maps per split")
    p.add_argument("--fig", type=Path, default=Path("results/figures/p0_samples.png"))
    p.add_argument("--no-fig", action="store_true", help="do not save the sample figure")
    args = p.parse_args(argv)

    meta = load_metadata(args.root)
    rep = Report()
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        check_split(rep, args.root, split, meta, args.max_maps)
        expected = meta.get("splits", {}).get(split, {}).get("samples_per_band")
        files = split_files(args.root, split)
        if expected and "index" in files and args.max_maps is None:
            n = len(read_index(files["index"]))
            rep.add("OK" if n == expected else "FAIL", f"{split} count",
                    f"{n} maps, metadata says {expected}")
    if not args.no_fig:
        fig_split = next((s for s in ("val", "train") if "pl58" in split_files(args.root, s)), None)
        if fig_split:
            save_figure(args.root, fig_split, read_index(split_files(args.root, fig_split)["index"]),
                        args.fig)
            rep.add("OK", "figure", f"saved {args.fig} ({fig_split})")
        else:
            rep.add("SKIP", "figure", "no split with pl58 files")
    print("\n".join(rep.lines))
    print("RESULT:", "FAIL" if rep.failed else "PASS")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
