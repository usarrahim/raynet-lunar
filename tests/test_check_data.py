"""Tests for raynet.check_data on a tiny fake dataset."""

import csv
import json
from pathlib import Path

import numpy as np

from raynet import check_data as cd

GRID = cd.GRID


def make_fake_split(root: Path, split: str, tx_list: list[tuple[str, int, int]]) -> None:
    """Write a fake split: one heightmap per terrain, PL grows with distance from the TX."""
    terrains = sorted({t for t, _, _ in tx_list})
    np.save(root / f"{split}_hm.npy", np.full((len(terrains), GRID, GRID), 5.0, np.float32))
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    pl, rows = [], []
    for i, (terrain, r, c) in enumerate(tx_list):
        pl.append(50.0 + 0.2 * np.hypot(yy - r, xx - c))
        rows.append({
            "sample_id": f"{terrain}_{i:02d}", "terrain_id": terrain, "tx_index": i,
            "group": "multi" if sum(t == terrain for t, _, _ in tx_list) > 1 else "single",
            "tx_row": r, "tx_col": c, "hm_row": terrains.index(terrain),
        })
    pl = np.stack(pl).astype(np.float16)
    mask = np.ones((len(tx_list), GRID, GRID), np.uint8)
    mask[:, :10, :10] = 0  # a small void in the corner
    packed = np.packbits(mask.reshape(len(tx_list), -1), axis=1)
    for b in cd.BANDS:
        pl_b = pl.copy()
        if b == "58":
            pl_b[:, :10, :10] = cd.FILL_58_DB
        np.save(root / f"{split}_pl{b}.npy", pl_b)
        np.save(root / f"{split}_mask{b}.npy", packed)
    with open(root / f"{split}_index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def make_fake_root(tmp_path: Path) -> Path:
    """Fake data root with metadata.json, a train split and a val split."""
    meta = {
        "heightmap_range_m": [0.0, 496.0],
        "pathloss_range_db_per_band": {"415": [20.0, 193.0], "58": [43.0, 228.0]},
        "splits": {"train": {"samples_per_band": 3}, "val": {"samples_per_band": 1}},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    make_fake_split(tmp_path, "train", [("00000", 50, 60), ("00000", 100, 20), ("00001", 128, 128)])
    make_fake_split(tmp_path, "val", [("00002", 30, 200)])
    return tmp_path


def test_unpack_masks_roundtrip() -> None:
    """Packing then unpacking a mask gives the same mask."""
    mask = np.random.default_rng(0).integers(0, 2, (2, GRID, GRID)).astype(bool)
    packed = np.packbits(mask.reshape(2, -1), axis=1)
    assert packed.shape == (2, 8192)
    assert np.array_equal(cd.unpack_masks(packed), mask)


def test_chunks_cover_range() -> None:
    """Chunks cover every index exactly once."""
    parts = cd.chunks(10, size=4)
    assert parts == [(0, 4), (4, 8), (8, 10)]


def test_band_stats_on_fake_data(tmp_path: Path) -> None:
    """Valid fraction, void share and near-TX minimum are computed correctly."""
    root = make_fake_root(tmp_path)
    rows = cd.read_index(root / "train_index.csv")
    st = cd.band_stats(root / "train_pl58.npy", root / "train_mask58.npy", rows)
    assert np.allclose(st["valid_frac"], 1 - 100 / GRID**2)
    assert st["invalid"] == 3 * 100
    assert st["invalid_fill"] == 3 * 100
    assert np.allclose(st["tx_min"], 50.0)
    assert np.allclose(st["map_min"], 50.0)


def test_main_passes_on_fake_data(tmp_path: Path, capsys) -> None:
    """The full check passes on good fake data and writes the figure."""
    root = make_fake_root(tmp_path)
    fig = tmp_path / "fig.png"
    assert cd.main(["--root", str(root), "--fig", str(fig)]) == 0
    out = capsys.readouterr().out
    assert "RESULT: PASS" in out
    assert "2 TX: 1 terrains" in out
    assert "[SKIP] test: no index file" in out
    assert fig.exists()


def test_main_fails_on_short_array(tmp_path: Path, capsys) -> None:
    """A PL array with fewer rows than the index makes the check fail."""
    root = make_fake_root(tmp_path)
    np.save(root / "train_pl415.npy", np.load(root / "train_pl415.npy")[:2])
    assert cd.main(["--root", str(root), "--split", "train", "--no-fig"]) == 1
    assert "[FAIL] train length" in capsys.readouterr().out
