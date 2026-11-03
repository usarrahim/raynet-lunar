"""Tests for the official-baseline wrapper (no official code needed)."""

import argparse
from types import SimpleNamespace

import pytest

from raynet.official_wrap import filter_samples, finish_official_args, terrains_of


def fake_ds() -> SimpleNamespace:
    """Dataset-like object: 3 terrains, terrain 'a' has 2 transmitters."""
    rows = [{"terrain_id": t} for t in ("a", "a", "b", "c")]
    return SimpleNamespace(rows=rows, samples=[(i, "58") for i in range(4)])


def test_filter_drop_and_keep_are_disjoint() -> None:
    """Dropping and keeping the same terrains splits the samples with no overlap."""
    train, dev = fake_ds(), fake_ds()
    assert filter_samples(train, {"a"}, keep=False) == 2
    assert filter_samples(dev, {"a"}, keep=True) == 2
    assert terrains_of(train) == {"b", "c"}
    assert terrains_of(dev) == {"a"}


def test_init_from_auto_is_refused() -> None:
    """'auto' would load the official _best.pt (chosen on val), so it is refused."""
    a = argparse.Namespace(run_name=None, band="58", loss="masked", phase="secondU",
                           init_from=None)
    with pytest.raises(SystemExit):
        finish_official_args(a)


def test_run_name_default() -> None:
    """The default run name follows the official pattern."""
    a = argparse.Namespace(run_name=None, band="58", loss="masked", phase="firstU",
                           init_from=None)
    assert finish_official_args(a).run_name == "radiownet_58_masked_firstU"
