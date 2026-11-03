"""Terrain-level dev split: a fixed 5% of the TRAINING terrains for tuning and decisions."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

DEV_FRACTION = 0.05
DEV_SEED = 0
DEV_FILE = Path("splits") / "dev_terrains.txt"


def terrain_groups(rows: list[dict[str, str]]) -> dict[str, str]:
    """Map terrain_id -> group ("multi" = 51 TX, "single" = 1 TX), from the TX count."""
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["terrain_id"]] += 1
    return {t: ("multi" if n > 1 else "single") for t, n in counts.items()}


def make_dev_split(
    rows: list[dict[str, str]], fraction: float = DEV_FRACTION, seed: int = DEV_SEED
) -> list[str]:
    """Pick `fraction` of the terrains of each group (multi / single); return sorted ids."""
    groups = terrain_groups(rows)
    rng = np.random.default_rng(seed)
    dev: list[str] = []
    for g in ("multi", "single"):
        ids = sorted(t for t, gg in groups.items() if gg == g)
        n = max(1, round(fraction * len(ids))) if ids else 0
        dev += [str(t) for t in rng.choice(ids, size=n, replace=False)]
    return sorted(dev)


def save_terrains(path: Path, terrains: list[str], header: str) -> None:
    """Write terrain ids, one per line, after '#' comment lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {h}" for h in header.splitlines()] + sorted(terrains)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_terrains(path: Path) -> set[str]:
    """Read terrain ids from a text file (skips empty lines and '#' comments)."""
    text = Path(path).read_text(encoding="utf-8")
    return {s.strip() for s in text.splitlines() if s.strip() and not s.startswith("#")}
