"""Build the results table of docs/04_EXPERIMENTS.md from val_summary.json files only.

Sources: results/eval/<ID>_<band>/val_summary.json (no training, for example E1, E1b) and
runs/<ID>_<band>_s<seed>/val_summary.json (trained models; mean ± std over seeds).
Missing results stay "TBD". Numbers are never typed by hand.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

ROWS = ("E0", "E1", "E1b", "E2", "E3", "E4", "E5a", "E5b", "E6")
NO_415 = {"E2", "E5a", "E5b"}  # planned for 5.8 GHz only
START, END = "<!-- results-table:start -->", "<!-- results-table:end -->"
RUN_RE = re.compile(r"^(?P<id>E\d+[a-z]?)_(?P<band>415|58)(_s(?P<seed>\d+))?$")


def find_summaries(roots: list[Path]) -> dict[tuple[str, str], list[dict]]:
    """Map (ID, band) -> list of val summaries (one per seed)."""
    found: dict[tuple[str, str], list[dict]] = {}
    for root in roots:
        for path in sorted(root.glob("*/val_summary.json")):
            m = RUN_RE.match(path.parent.name)
            if m:
                s = json.loads(path.read_text(encoding="utf-8"))
                if not s.get("smoke_test"):
                    found.setdefault((m["id"], m["band"]), []).append(s)
    return found


def mean_std(values: list[float]) -> str:
    """'x.xx' for one value, 'x.xx ± y.yy' for several."""
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"


def cell(runs: list[dict] | None, key: str | None = None, region: str | None = None) -> str:
    """One table cell from the runs of one (ID, band)."""
    if not runs:
        return "TBD"
    if region:
        return mean_std([r["regions"][region]["rmse_db"] for r in runs])
    return mean_std([r[key or "rmse_db"] for r in runs])


def build_table(found: dict[tuple[str, str], list[dict]]) -> str:
    """Markdown table, one row per experiment ID."""
    header = ("| ID | 415 MHz RMSE (dB) | 5.8 GHz RMSE (dB) | Combined | NLOS core 5.8 | "
              "Border 5.8 | Params (per band) | Seeds |")
    lines = [header, "|---|---|---|---|---|---|---|---|"]
    for rid in ROWS:
        r415, r58 = found.get((rid, "415")), found.get((rid, "58"))
        c415 = "-" if rid in NO_415 and not r415 else cell(r415)
        if r415 and r58:
            m415 = statistics.mean(r["rmse_db"] for r in r415)
            m58 = statistics.mean(r["rmse_db"] for r in r58)
            comb = f"{0.5 * m415 + 0.5 * m58:.2f}"
        else:
            comb = "-" if rid in NO_415 else "TBD"
        params = next((str(r["params"]) for r in (r58 or r415 or []) if "params" in r), "TBD")
        seeds = str(len(r58)) if r58 else "-"
        lines.append(f"| {rid} | {c415} | {cell(r58)} | {comb} | "
                     f"{cell(r58, region='nlos_core')} | {cell(r58, region='border')} | "
                     f"{params} | {seeds} |")
    return "\n".join(lines)


def update_doc(doc: Path, table: str) -> None:
    """Replace the text between the START and END markers in the doc."""
    text = doc.read_text(encoding="utf-8")
    i, j = text.index(START) + len(START), text.index(END)
    doc.write_text(text[:i] + "\n" + table + "\n" + text[j:], encoding="utf-8")
