"""Dataset for the LunarRM flat-array files (docs/01_DATA.md, loader plan).

One item is a dict with raw units (metres, dB). Normalization happens in the model wrapper.
Arrays are opened lazily with memory maps, so each DataLoader worker opens its own copy.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from raynet.check_data import read_index
from raynet.splits import DEV_FILE, load_terrains

GRID = 256
BANDS = ("415", "58")


def dihedral(a: np.ndarray, k: int) -> np.ndarray:
    """Apply dihedral transform k (0..7) to the last two axes: optional left-right flip, then k%4 x rot90."""
    if k >= 4:
        a = a[..., :, ::-1]
    return np.ascontiguousarray(np.rot90(a, k % 4, axes=(-2, -1)))


def dihedral_point(row: int, col: int, k: int, n: int = GRID) -> tuple[int, int]:
    """Where pixel (row, col) goes under `dihedral(., k)` on an n x n grid."""
    if k >= 4:
        col = n - 1 - col
    for _ in range(k % 4):  # np.rot90 (counter-clockwise): (r, c) -> (n-1-c, r)
        row, col = n - 1 - col, row
    return row, col


class LunarRMDataset(Dataset):
    """Maps of one band from one split, optionally filtered by terrain."""

    def __init__(
        self,
        root: Path | str,
        split: str,
        band: str,
        terrains: set[str] | None = None,
        exclude: set[str] | None = None,
        augment: bool = False,
        max_items: int | None = None,
    ) -> None:
        """Read the index and keep the rows of the wanted terrains."""
        if band not in BANDS:
            raise ValueError(f"band must be one of {BANDS}, got {band!r}")
        self.root, self.split, self.band, self.augment = Path(root), split, band, augment
        rows = read_index(self.root / f"{split}_index.csv")
        items = [
            (i, r) for i, r in enumerate(rows)
            if (terrains is None or r["terrain_id"] in terrains)
            and (exclude is None or r["terrain_id"] not in exclude)
        ]
        self.items = items[:max_items] if max_items else items
        self._arrays: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        """Number of maps."""
        return len(self.items)

    def __getstate__(self) -> dict:
        """Do not send open memory maps to worker processes (Windows spawn copies them)."""
        state = self.__dict__.copy()
        state["_arrays"] = None
        return state

    def _open(self) -> dict[str, np.ndarray]:
        """Open the arrays of this split and band as memory maps (once per process)."""
        if self._arrays is None:
            name = {"hm": "hm", "pl": f"pl{self.band}", "mask": f"mask{self.band}"}
            self._arrays = {
                k: np.load(self.root / f"{self.split}_{v}.npy", mmap_mode="r")
                for k, v in name.items()
            }
        return self._arrays

    def terrain_ids(self) -> set[str]:
        """Set of terrain ids in this dataset."""
        return {r["terrain_id"] for _, r in self.items}

    def get(self, idx: int, k: int = 0) -> dict:
        """Item `idx` with dihedral transform k (0 = no change)."""
        arr = self._open()
        i, row = self.items[idx]
        height = np.array(arr["hm"][int(row["hm_row"])], dtype=np.float32)  # copy, writable
        pl = np.array(arr["pl"][i], dtype=np.float32)
        mask = np.unpackbits(np.asarray(arr["mask"][i])).reshape(GRID, GRID).astype(np.float32)
        tr, tc = int(row["tx_row"]), int(row["tx_col"])
        if k:
            height, pl, mask = dihedral(height, k), dihedral(pl, k), dihedral(mask, k)
            tr, tc = dihedral_point(tr, tc, k)
        return {
            "height": torch.from_numpy(height)[None],
            "tx": torch.tensor([tr, tc], dtype=torch.float32),
            "pl": torch.from_numpy(pl)[None],
            "mask": torch.from_numpy(mask)[None],
            "band": self.band,
            "sample_id": row["sample_id"],
            "terrain_id": row["terrain_id"],
        }

    def __getitem__(self, idx: int) -> dict:
        """Item `idx`; a random dihedral transform when augment is on."""
        return self.get(idx, random.randrange(8) if self.augment else 0)


def make_datasets(
    root: Path | str, band: str, dev_file: Path = DEV_FILE, augment: bool = True
) -> dict[str, LunarRMDataset]:
    """The three datasets: train (dev terrains removed), dev, and the frozen val."""
    dev = load_terrains(dev_file)
    return {
        "train": LunarRMDataset(root, "train", band, exclude=dev, augment=augment),
        "dev": LunarRMDataset(root, "train", band, terrains=dev),
        "val": LunarRMDataset(root, "val", band),
    }
