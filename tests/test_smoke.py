"""Smoke test: the package is installed and can be imported."""

import raynet


def test_import_and_version() -> None:
    """The package imports and has a version string."""
    assert isinstance(raynet.__version__, str)
    assert raynet.__version__.count(".") == 2
