"""Tests for the results table builder."""

import json
from pathlib import Path

from raynet.results_table import END, START, build_table, find_summaries, update_doc


def write_summary(folder: Path, rmse: float, smoke: bool = False) -> None:
    """Write a minimal val_summary.json."""
    folder.mkdir(parents=True)
    regions = {k: {"rmse_db": rmse + 1} for k in ("los_core", "border", "nlos_core")}
    folder.joinpath("val_summary.json").write_text(json.dumps(
        {"rmse_db": rmse, "regions": regions, "params": 5, "smoke_test": smoke}))


def test_table_from_files(tmp_path: Path) -> None:
    """Seeds give mean ± std; missing results are TBD; smoke tests are ignored."""
    write_summary(tmp_path / "E1b_415", 5.0)
    write_summary(tmp_path / "E1b_58", 7.0)
    write_summary(tmp_path / "E3_58_s0", 4.0)
    write_summary(tmp_path / "E3_58_s1", 6.0)
    write_summary(tmp_path / "E4_58_s0", 1.0, smoke=True)
    table = build_table(find_summaries([tmp_path]))
    rows = {line.split("|")[1].strip(): line for line in table.splitlines()[2:]}
    assert "| 5.00 | 7.00 | 6.00 |" in rows["E1b"]
    assert "5.00 ± 1.41" in rows["E3"]
    assert "TBD" in rows["E4"]
    assert rows["E2"].split("|")[2].strip() == "-"


def test_update_doc_keeps_other_text(tmp_path: Path) -> None:
    """Only the text between the markers changes."""
    doc = tmp_path / "doc.md"
    doc.write_text(f"before\n{START}\nold\n{END}\nafter\n", encoding="utf-8")
    update_doc(doc, "new table")
    assert doc.read_text(encoding="utf-8") == f"before\n{START}\nnew table\n{END}\nafter\n"
