"""Write a region / distance report from evaluation summaries; the FIRST file is the baseline for Gate G1.

Run: python scripts/region_report.py results/eval/P16_58_s0_firstU/dev_summary.json
     results/eval/E1b_58/dev_summary.json --out results/p1_error_regions.md
"""

import argparse
from pathlib import Path

from raynet.region_report import distance_table, gate_g1, load, region_table

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summaries", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default="Error by region and distance")
    args = p.parse_args()
    sums = [load(s) for s in args.summaries]
    text = "\n\n".join([
        f"# {args.title}",
        "Made by `scripts/region_report.py` from: " + ", ".join(f"`{s}`" for s in args.summaries),
        "Each cell: RMSE (dB) / share of valid pixels (%) / share of the squared error (%).",
        region_table(sums),
        "## RMSE (dB) by distance from the TX",
        distance_table(sums),
        "## Gate G1",
        gate_g1(sums[0]),
    ]) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
