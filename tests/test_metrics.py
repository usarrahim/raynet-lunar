"""Tests for raynet.metrics with small hand-made arrays (known answers)."""

import math

import torch

from raynet.metrics import (
    MetricAccumulator,
    boundary,
    combined_score,
    disk_kernel,
    distance_bin_masks,
    region_masks,
)


def maps(*values: float, size: int = 4) -> torch.Tensor:
    """Batch of constant maps (B, 1, size, size)."""
    return torch.stack([torch.full((1, size, size), float(v)) for v in values])


def test_masked_rmse_ignores_invalid_pixels() -> None:
    """Error 2 dB on valid pixels, huge error on invalid pixels → RMSE is 2."""
    gt = maps(100.0)
    pred = gt + 2.0
    mask = torch.ones_like(gt)
    mask[0, 0, 0, :] = 0
    pred[0, 0, 0, :] = 999.0
    acc = MetricAccumulator()
    acc.update(pred, gt, mask)
    s = acc.summary()
    assert math.isclose(s["rmse_db"], 2.0)
    assert math.isclose(s["bias_db"], 2.0)
    assert s["n_valid_px"] == 12
    assert s["rmse_db_allpx_diagnostic"] > 100


def test_pooled_differs_from_mean_per_map() -> None:
    """Pooled RMSE weights maps by valid pixels; the mean of per-map RMSE does not."""
    gt = maps(0.0, 0.0)
    pred = maps(1.0, 3.0)
    mask = torch.ones_like(gt)
    mask[1, 0, :, 1:] = 0  # map 2 has only 4 valid pixels
    acc = MetricAccumulator()
    acc.update(pred, gt, mask)
    s = acc.summary()
    assert math.isclose(s["rmse_db"], math.sqrt((16 * 1 + 4 * 9) / 20))
    assert math.isclose(s["mean_per_map_rmse_db"], 2.0)
    assert [r["rmse_all"] for r in acc.rows] == [1.0, 3.0]


def test_boundary_and_border_width() -> None:
    """A vertical LoS edge: border covers 3 px on each side plus the edge pixels."""
    los = torch.zeros(1, 1, 20, 20)
    los[..., :, :10] = 1  # columns 0-9 LoS, 10-19 NLOS
    b = boundary(los)
    assert b[0, 0, 5].nonzero().flatten().tolist() == [9, 10]
    reg = region_masks(los, torch.ones_like(los))
    border_cols = reg["border"][0, 0, 5].nonzero().flatten().tolist()
    assert border_cols == list(range(6, 14))
    total = sum(m.float() for m in reg.values())
    assert torch.all(total == 1)  # disjoint and complete inside the valid mask


def test_disk_kernel_radius_3() -> None:
    """Euclidean disk of radius 3 has 29 pixels."""
    assert int(disk_kernel(3).sum()) == 29


def test_distance_bins_cover_valid_pixels_once() -> None:
    """Every valid pixel is in exactly one distance bin."""
    tx = torch.tensor([[0.0, 0.0]])
    valid = torch.ones(1, 1, 256, 256, dtype=torch.bool)
    bins = distance_bin_masks(tx, valid)
    total = sum(m.float() for m in bins.values())
    assert torch.all(total == 1)
    assert bins["0-25m"][0, 0, 0, 24] and not bins["0-25m"][0, 0, 0, 25]


def test_coverage_accuracy_and_f1() -> None:
    """Threshold 120 dB: 2 of 4 pixels right; F1 of 'not covered' from tp/fp/fn."""
    gt = torch.tensor([110.0, 130.0, 130.0, 110.0]).reshape(1, 1, 2, 2)
    pred = torch.tensor([110.0, 130.0, 110.0, 130.0]).reshape(1, 1, 2, 2)
    acc = MetricAccumulator()
    acc.update(pred, gt, torch.ones_like(gt))
    cov = acc.summary()["coverage"]["120"]
    assert math.isclose(cov["accuracy"], 0.5)
    assert math.isclose(cov["f1_not_covered"], 2 * 1 / (2 * 1 + 1 + 1))


def test_regions_in_summary() -> None:
    """Region RMSE and SSE shares are reported when LoS is given."""
    gt = torch.zeros(1, 1, 20, 20)
    pred = gt.clone()
    pred[..., :, 15:] = 4.0  # error only deep in the NLOS side
    los = torch.zeros_like(gt)
    los[..., :, :10] = 1
    acc = MetricAccumulator()
    acc.update(pred, gt, torch.ones_like(gt), los=los, tx=torch.tensor([[10.0, 0.0]]))
    s = acc.summary()
    assert math.isclose(s["regions"]["nlos_core"]["sse_share"], 1.0)
    assert s["regions"]["los_core"]["rmse_db"] == 0.0
    assert "distance_bins" in s


def test_combined_score() -> None:
    """Combined score is the mean of the two bands."""
    assert combined_score(4.0, 6.0) == 5.0
