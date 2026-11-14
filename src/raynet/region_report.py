"""Region and distance report from evaluation summaries (Gate G1, docs/08_ROADMAP.md P1.6).

Gate G1: if nlos_core + border hold less than 40% of the squared error of the baseline,
stop and discuss the method focus (the HW1 "weakest assumption" check).
"""

from __future__ import annotations

import json
from pathlib import Path

from raynet.metrics import REGIONS

G1_MIN_SHARE = 0.40


def load(path: Path | str) -> dict:
    """Read one <split>_summary.json."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def shadow_share(summary: dict) -> float:
    """Share of the squared error in nlos_core + border."""
    reg = summary["regions"]
    return reg["nlos_core"]["sse_share"] + reg["border"]["sse_share"]


def region_table(summaries: list[dict]) -> str:
    """Markdown table: one row per model, RMSE and SSE share per region."""
    head = "| Model | Split | RMSE all | " + " | ".join(
        f"{r} RMSE / px % / SSE %" for r in REGIONS) + " | nlos+border SSE % |"
    lines = [head, "|" + "---|" * (len(REGIONS) + 4)]
    for s in summaries:
        cells = [f"{s['regions'][r]['rmse_db']:.2f} / {100 * s['regions'][r]['pixel_share']:.1f}"
                 f" / {100 * s['regions'][r]['sse_share']:.1f}" for r in REGIONS]
        lines.append(f"| {s['model']} | {s['split']} | {s['rmse_db']:.2f} | " + " | ".join(cells)
                     + f" | {100 * shadow_share(s):.1f} |")
    return "\n".join(lines)


def distance_table(summaries: list[dict]) -> str:
    """Markdown table: RMSE per distance bin, one row per model."""
    bins = list(summaries[0]["distance_bins"])
    lines = ["| Model | " + " | ".join(bins) + " |", "|" + "---|" * (len(bins) + 1)]
    for s in summaries:
        lines.append(f"| {s['model']} | " + " | ".join(
            f"{s['distance_bins'][b]['rmse_db']:.2f}" for b in bins) + " |")
    return "\n".join(lines)


def gate_g1(baseline: dict) -> str:
    """Gate G1 verdict text for the baseline summary."""
    share = shadow_share(baseline)
    verdict = "PASS" if share >= G1_MIN_SHARE else "STOP (discuss the method focus)"
    return (f"Gate G1 ({baseline['model']}, {baseline['split']}): nlos_core + border hold "
            f"{100 * share:.1f}% of the squared error (threshold {100 * G1_MIN_SHARE:.0f}%) -> "
            f"**{verdict}**")
