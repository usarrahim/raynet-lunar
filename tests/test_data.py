"""Tests for raynet.data (fake data, plus a few real items if data/LunarRM exists)."""

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from test_check_data import make_fake_root

from raynet.data import LunarRMDataset, dihedral, dihedral_point, make_datasets

REAL_ROOT = Path("data/LunarRM")


def test_dihedral_point_matches_array_transform() -> None:
    """A marked pixel lands where dihedral_point says, for all 8 transforms."""
    a = np.zeros((256, 256))
    a[10, 200] = 1.0
    seen = set()
    for k in range(8):
        r, c = dihedral_point(10, 200, k)
        b = dihedral(a, k)
        assert b[r, c] == 1.0 and b.sum() == 1.0
        seen.add((r, c))
    assert len(seen) == 8  # all 8 transforms are different


def test_item_shapes_and_types(tmp_path: Path) -> None:
    """Item fields have the shapes and types of docs/01_DATA.md."""
    root = make_fake_root(tmp_path)
    item = LunarRMDataset(root, "train", "58")[0]
    assert item["height"].shape == (1, 256, 256) and item["height"].dtype == torch.float32
    assert item["pl"].shape == (1, 256, 256) and item["mask"].shape == (1, 256, 256)
    assert item["tx"].tolist() == [50.0, 60.0]
    assert set(item["mask"].unique().tolist()) <= {0.0, 1.0}
    assert item["band"] == "58" and item["terrain_id"] == "00000"


def test_augment_keeps_tx_aligned(tmp_path: Path) -> None:
    """After every transform the minimum PL is still at the TX pixel."""
    root = make_fake_root(tmp_path)
    ds = LunarRMDataset(root, "train", "415")
    for k in range(8):
        item = ds.get(2, k)
        r, c = (int(v) for v in item["tx"])
        pl = item["pl"][0]
        assert pl[r, c] == pl.min()


def test_terrain_filters(tmp_path: Path) -> None:
    """`terrains` keeps only those terrains, `exclude` removes them."""
    root = make_fake_root(tmp_path)
    keep = LunarRMDataset(root, "train", "58", terrains={"00000"})
    drop = LunarRMDataset(root, "train", "58", exclude={"00000"})
    assert len(keep) == 2 and keep.terrain_ids() == {"00000"}
    assert len(drop) == 1 and drop.terrain_ids() == {"00001"}


def test_pickle_drops_open_arrays(tmp_path: Path) -> None:
    """A pickled dataset (as sent to a worker) does not carry the memory maps."""
    ds = LunarRMDataset(make_fake_root(tmp_path), "train", "58")
    ds[0]
    assert ds._arrays is not None
    assert pickle.loads(pickle.dumps(ds))._arrays is None


@pytest.mark.skipif(not (REAL_ROOT / "train_index.csv").exists(), reason="no local data")
def test_real_data_split_and_tx() -> None:
    """Real data: dev terrains are not in train; height at the TX is the same after a transform."""
    sets = make_datasets(REAL_ROOT, "58")
    assert not (sets["train"].terrain_ids() & sets["dev"].terrain_ids())
    assert len(sets["train"]) + len(sets["dev"]) == 25720
    assert len(sets["val"]) == 2330
    ds = sets["dev"]
    for k in range(8):
        a, b = ds.get(0, 0), ds.get(0, k)
        ra, ca = (int(v) for v in a["tx"])
        rb, cb = (int(v) for v in b["tx"])
        assert a["height"][0, ra, ca] == b["height"][0, rb, cb]
        assert a["pl"][0, ra, ca] == b["pl"][0, rb, cb]
