"""Wrap a checkpoint of the OFFICIAL baseline (RadioWNet) so our evaluation can use it (E0, P1.6).

The official code is not copied into our package: we import it from a clone of
https://github.com/anderspearson206/lunar-radiomap-challenge (commit fa3ef9b).
Its conventions (from its lunar_dataset.py): input = [height normalized with heightmap_range_m,
TX one-hot]; target = pathloss normalized with the GLOBAL pathloss_range_db and clipped to [0, 1].
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import torch
from torch import nn


class OfficialRanges:
    """Normalization of the official loader (global ranges from metadata.json)."""

    def __init__(self, data_root: Path | str) -> None:
        """Read heightmap_range_m and pathloss_range_db."""
        meta = json.loads((Path(data_root) / "metadata.json").read_text(encoding="utf-8"))
        self.hm_min, self.hm_max = (float(v) for v in meta["heightmap_range_m"])
        self.pl_min, self.pl_max = (float(v) for v in meta["pathloss_range_db"])


class OfficialWrapper(nn.Module):
    """(height m, tx (row, col)) -> pathloss dB with an official RadioWNet inside."""

    def __init__(self, net: nn.Module, out_index: int, ranges: OfficialRanges) -> None:
        """out_index 0 = first U-Net (firstU stage), 1 = refined output (secondU stage)."""
        super().__init__()
        self.net, self.out_index, self.r = net, out_index, ranges

    def inputs(self, height: torch.Tensor, tx: torch.Tensor) -> torch.Tensor:
        """Official input (B, 2, H, W): normalized clipped height and TX one-hot."""
        b, _, h, w = height.shape
        hm = ((height - self.r.hm_min) / (self.r.hm_max - self.r.hm_min)).clamp(0, 1)
        rows = torch.arange(h, device=height.device).view(1, 1, h, 1)
        cols = torch.arange(w, device=height.device).view(1, 1, 1, w)
        onehot = (rows == tx[:, 0].round().long().view(b, 1, 1, 1)) \
            & (cols == tx[:, 1].round().long().view(b, 1, 1, 1))
        return torch.cat([hm, onehot.float()], dim=1)

    def forward(self, height: torch.Tensor, tx: torch.Tensor) -> torch.Tensor:
        """Pathloss in dB (B, 1, H, W)."""
        y = self.net(self.inputs(height.float(), tx.float()))[self.out_index].float()
        return y * (self.r.pl_max - self.r.pl_min) + self.r.pl_min


def import_official_radiownet(official_repo: Path | str) -> type:
    """Import RadioWNet from RadioUNet/modules.py of the official clone."""
    path = Path(official_repo) / "RadioUNet" / "modules.py"
    if "torchvision" not in sys.modules:
        try:
            import torchvision  # noqa: F401
        except ImportError:  # modules.py imports torchvision but never uses it
            stub = types.ModuleType("torchvision")
            stub.models = types.ModuleType("torchvision.models")
            sys.modules["torchvision"], sys.modules["torchvision.models"] = stub, stub.models
    spec = importlib.util.spec_from_file_location("official_radiounet_modules", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RadioWNet


def load_official(ckpt_path: Path | str, official_repo: Path | str,
                  data_root: Path | str, device: str = "cpu") -> tuple[OfficialWrapper, dict]:
    """Official checkpoint (*_last.pt) -> wrapper in eval mode, plus the checkpoint args."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ck.get("args", {})
    if args.get("band") not in ("415", "58"):
        raise SystemExit("only single-band official checkpoints are supported")
    radiownet = import_official_radiownet(official_repo)
    net = radiownet(inputs=2, phase=args.get("phase", "firstU"))
    net.load_state_dict(ck["model"])
    out_index = 0 if args.get("phase", "firstU") == "firstU" else 1
    model = OfficialWrapper(net, out_index, OfficialRanges(data_root)).to(device).eval()
    return model, {**args, "epoch": ck.get("epoch")}
