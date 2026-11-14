"""Tests for the region report and Gate G1."""

from raynet.region_report import distance_table, gate_g1, region_table, shadow_share


def summary(nlos: float, border: float) -> dict:
    """Minimal summary with given SSE shares."""
    regions = {
        "los_core": {"rmse_db": 3.0, "pixel_share": 0.8, "sse_share": 1 - nlos - border},
        "border": {"rmse_db": 9.0, "pixel_share": 0.1, "sse_share": border},
        "nlos_core": {"rmse_db": 8.0, "pixel_share": 0.1, "sse_share": nlos},
    }
    bins = {"0-25m": {"rmse_db": 1.0}, "25-50m": {"rmse_db": 2.0}}
    return {"model": "M", "split": "dev", "rmse_db": 5.0, "regions": regions,
            "distance_bins": bins}


def test_gate_g1_threshold() -> None:
    """40% or more in nlos + border passes; less stops."""
    assert abs(shadow_share(summary(0.3, 0.2)) - 0.5) < 1e-9
    assert "PASS" in gate_g1(summary(0.3, 0.2))
    assert "STOP" in gate_g1(summary(0.1, 0.2))


def test_tables_have_one_row_per_model() -> None:
    """Header, separator and one row per summary."""
    s = [summary(0.3, 0.2), summary(0.1, 0.1)]
    assert len(region_table(s).splitlines()) == 4
    assert "| M | 1.00 | 2.00 |" in distance_table(s)
