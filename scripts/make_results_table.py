"""Rebuild the results table in docs/04_EXPERIMENTS.md from val_summary.json files.

Run: python scripts/make_results_table.py
"""

from pathlib import Path

from raynet.results_table import build_table, find_summaries, update_doc

if __name__ == "__main__":
    table = build_table(find_summaries([Path("results/eval"), Path("runs")]))
    update_doc(Path("docs/04_EXPERIMENTS.md"), table)
    print(table)
