"""Tests for the official-baseline wrapper (a dummy network stands in for RadioWNet)."""

import json
from pathlib import Path

import torch
from torch import nn

from raynet.models.radiounet_wrap import OfficialRanges, OfficialWrapper


class Echo(nn.Module):
    """Returns [input channel 0, input channel 1] like RadioWNet returns [out1, out2]."""

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Two outputs."""
        return [x[:, 0:1], x[:, 1:2]]


def ranges(tmp_path: Path) -> OfficialRanges:
    """Ranges from a small metadata.json."""
    (tmp_path / "metadata.json").write_text(json.dumps(
        {"heightmap_range_m": [0.0, 400.0], "pathloss_range_db": [20.0, 228.0]}))
    return OfficialRanges(tmp_path)


def test_inputs_match_official_convention(tmp_path: Path) -> None:
    """Height is normalized and clipped; the TX one-hot has one pixel at (row, col)."""
    wrap = OfficialWrapper(Echo(), 0, ranges(tmp_path))
    h = torch.full((1, 1, 8, 8), 200.0)
    h[0, 0, 0, 0] = 999.0
    x = wrap.inputs(h, torch.tensor([[2.0, 5.0]]))
    assert torch.isclose(x[0, 0, 1, 1], torch.tensor(0.5)) and x[0, 0, 0, 0] == 1.0
    assert x[0, 1].sum() == 1 and x[0, 1, 2, 5] == 1


def test_output_index_and_denormalization(tmp_path: Path) -> None:
    """Output 0 is the first U-Net; normalized 0.5 becomes 124 dB with the global range."""
    h = torch.full((1, 1, 8, 8), 200.0)
    tx = torch.tensor([[2.0, 5.0]])
    first = OfficialWrapper(Echo(), 0, ranges(tmp_path))(h, tx)
    second = OfficialWrapper(Echo(), 1, ranges(tmp_path))(h, tx)
    assert torch.allclose(first, torch.full_like(first, 124.0))
    assert second[0, 0, 2, 5] == 228.0 and second[0, 0, 0, 0] == 20.0
