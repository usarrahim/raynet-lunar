"""Tests for the terrain-level dev split."""

from pathlib import Path

from raynet.splits import load_terrains, make_dev_split, save_terrains, terrain_groups


def fake_rows() -> list[dict[str, str]]:
    """20 terrains with 51 TX and 400 terrains with 1 TX."""
    rows = [{"terrain_id": f"m{t:03d}"} for t in range(20) for _ in range(51)]
    rows += [{"terrain_id": f"s{t:04d}"} for t in range(400)]
    return rows


def test_groups_from_tx_count() -> None:
    """Terrains with more than one TX are 'multi'."""
    g = terrain_groups(fake_rows())
    assert g["m000"] == "multi" and g["s0000"] == "single"


def test_dev_split_is_stratified_and_fixed() -> None:
    """5% of each group, same result for the same seed, other result for another seed."""
    rows = fake_rows()
    dev = make_dev_split(rows, 0.05, seed=0)
    assert sum(t.startswith("m") for t in dev) == 1
    assert sum(t.startswith("s") for t in dev) == 20
    assert dev == make_dev_split(rows, 0.05, seed=0)
    assert dev != make_dev_split(rows, 0.05, seed=1)


def test_all_tx_of_a_terrain_stay_together() -> None:
    """A terrain is either fully in dev or fully out (no split of its transmitters)."""
    rows = fake_rows()
    dev = set(make_dev_split(rows, 0.05, seed=0))
    in_dev = [r["terrain_id"] in dev for r in rows if r["terrain_id"] == next(iter(dev))]
    assert all(in_dev)


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    """Saved ids come back the same; comment lines are skipped."""
    path = tmp_path / "dev.txt"
    save_terrains(path, ["00002", "00001"], "line one\nline two")
    assert load_terrains(path) == {"00001", "00002"}
