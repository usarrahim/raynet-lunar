"""Small helpers to log where a result came from (git commit, device)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import torch


def git_commit(path: Path | str = ".") -> str:
    """Short git commit hash of a folder (with '+dirty' if there are local changes)."""
    try:
        head = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain",
                                "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def device_name(device: str) -> str:
    """GPU name for cuda devices, else 'cpu'."""
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.device(device))
    return "cpu"
