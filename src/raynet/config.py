"""YAML configs: an experiment file can name a `base` file and override some keys."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Copy of `base` with `override` merged in (nested dicts are merged, not replaced)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: Path | str) -> dict:
    """Read a YAML config; `base: other.yaml` (relative to this file) is loaded first."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = deep_merge(load_config(path.parent / base), cfg)
    return cfg


def save_config(cfg: dict, path: Path) -> None:
    """Write the fully resolved config as YAML."""
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
