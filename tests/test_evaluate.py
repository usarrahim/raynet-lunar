"""Tests for the prior models, the E1b fit and the evaluation script (fake data)."""

import json
from pathlib import Path

import numpy as np
import torch
from test_check_data import make_fake_root

from raynet import calibrate, evaluate
from raynet.data import LunarRMDataset
from raynet.models.prior import PriorModel, e1b_design
from raynet.physics import BAND_HZ


def fake_root_with_dev(tmp_path: Path) -> tuple[Path, Path]:
    """Fake data root plus a dev file that holds one training terrain."""
    root = make_fake_root(tmp_path)
    dev = tmp_path / "dev.txt"
    dev.write_text("# dev\n00000\n", encoding="utf-8")
    return root, dev


def test_e1_prior_on_flat_terrain_is_fspl() -> None:
    """E1 on flat terrain equals FSPL (J = 0 everywhere)."""
    h = torch.zeros(1, 1, 256, 256)
    tx = torch.tensor([[128.0, 128.0]])
    pl, feats = PriorModel().forward_with_features(h, tx, torch.tensor([BAND_HZ["58"]]))
    assert torch.allclose(pl, feats[:, 0:1])


def test_e1b_recovers_known_coefficients(tmp_path: Path) -> None:
    """If PL is an exact linear mix of the terms, the fit finds the same 5 numbers."""
    root, _ = fake_root_with_dev(tmp_path)
    hm = np.load(root / "train_hm.npy")
    hm[:, :, 150:153] = 20.0  # a wall, so LoS, J and shadow are not constant
    hm[:, 30:33, :] = 15.0
    np.save(root / "train_hm.npy", hm)
    ds = LunarRMDataset(root, "train", "58")
    true = [5.0, 1.2, 0.5, -3.0, 0.8]
    model = PriorModel(true)
    arr = np.array(np.load(root / "train_pl58.npy"))
    for i in range(len(ds)):  # replace the targets by the E1b formula
        item = ds.get(i)
        pl = model(item["height"][None], item["tx"][None], torch.tensor([BAND_HZ["58"]]))
        arr[ds.items[i][0]] = pl[0, 0].numpy().astype(np.float16)
    ds._arrays = None  # close the memory maps before writing (Windows)
    np.save(root / "train_pl58.npy", arr)
    ds = LunarRMDataset(root, "train", "58")
    fit = calibrate.fit_e1b(ds, "58")
    # float16 targets: about 0.03 dB rounding, so the fit is close, not exact
    assert np.allclose(fit["coeffs"], true, atol=0.05), fit["coeffs"]
    assert fit["fit_rmse_db"] < 0.1


def test_design_matrix_columns() -> None:
    """Design matrix has the constant 1 first and shadow in degrees last."""
    feats = torch.zeros(1, 6, 2, 2)
    feats[:, 2] = np.pi / 180  # 1 degree of shadow
    x = e1b_design(feats)
    assert torch.all(x[:, 0] == 1) and torch.allclose(x[:, 4], torch.ones(1, 2, 2))


def test_evaluate_cli_writes_files(tmp_path: Path) -> None:
    """The CLI writes per-map CSV and summary JSON with the expected keys."""
    root, dev = fake_root_with_dev(tmp_path)
    out = tmp_path / "eval"
    s = evaluate.main(["--data-root", str(root), "--model", "e1", "--band", "58",
                       "--split", "dev", "--dev-file", str(dev), "--out", str(out),
                       "--device", "cpu"])
    assert (out / "dev_per_map.csv").exists()
    saved = json.loads((out / "dev_summary.json").read_text(encoding="utf-8"))
    assert saved["model"] == "E1" and saved["n_maps"] == s["n_maps"] == 2
    assert set(saved["regions"]) == {"los_core", "border", "nlos_core"}
    assert "120" in saved["coverage"]
