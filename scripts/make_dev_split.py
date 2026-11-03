"""Make splits/dev_terrains.txt: 5% of the training terrains (per group), seed 0.

Run: python scripts/make_dev_split.py --data-root data/LunarRM
"""

import argparse
from collections import Counter
from pathlib import Path

from raynet.check_data import read_index
from raynet.splits import (
    DEV_FILE,
    DEV_FRACTION,
    DEV_SEED,
    make_dev_split,
    save_terrains,
    terrain_groups,
)


def main() -> None:
    """Build the dev split from train_index.csv and print a short summary."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=DEV_FILE)
    args = p.parse_args()

    rows = read_index(args.data_root / "train_index.csv")
    dev = make_dev_split(rows, DEV_FRACTION, DEV_SEED)
    groups = terrain_groups(rows)
    dev_groups = Counter(groups[t] for t in dev)
    all_groups = Counter(groups.values())
    n_maps = sum(1 for r in rows if r["terrain_id"] in set(dev))
    header = (f"Dev split: {DEV_FRACTION:.0%} of TRAINING terrains per group, seed {DEV_SEED}.\n"
              f"Made by scripts/make_dev_split.py. Tuning and decisions only; never train on these.\n"
              f"multi (51 TX): {dev_groups['multi']} of {all_groups['multi']} terrains; "
              f"single (1 TX): {dev_groups['single']} of {all_groups['single']}; "
              f"{n_maps} maps per band.")
    save_terrains(args.out, dev, header)
    print(header)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
