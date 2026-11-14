"""Tests for the RayNet polar branch and the E4 / E5a models (docs/08_ROADMAP.md P3.1)."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from raynet.models.prior import PriorModel
from raynet.models.raynet import POLAR_CHANNELS, PolarBranch, RayScan
from raynet.models.unet import count_params
from raynet.models.wrapper import PathlossNet, Ranges
from raynet.physics import BAND_HZ, PhysicsFeatures

RANGES = Ranges(0.0, 496.0, 43.0, 228.0)
COEFFS = [15.9, 1.03, 2.3, -18.3, -9.9]


def bumpy(b: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Smooth random terrain batch and TX positions."""
    torch.manual_seed(0)
    h = torch.nn.functional.avg_pool2d(torch.rand(b, 1, 256, 256) * 400, 31, 1, 15)
    tx = torch.tensor([[40.0, 200.0], [128.0, 7.0]])[:b]
    return h, tx


def test_polar_branch_shapes() -> None:
    """7 polar inputs on (T, R) = (360, 256); output back on the 256 x 256 grid."""
    h, tx = bumpy()
    branch = PolarBranch()
    x = branch.polar_inputs(h, tx, torch.full((2,), BAND_HZ["58"]))
    assert x.shape == (2, len(POLAR_CHANNELS), 360, 256) and torch.isfinite(x).all()
    assert branch(h, tx, torch.full((2,), BAND_HZ["58"])).shape == (2, 32, 256, 256)


def test_cartesian_polar_round_trip() -> None:
    """A smooth image survives Cartesian -> polar -> Cartesian with a small error."""
    pf = PhysicsFeatures(n_theta=360, n_r=256)
    yy, xx = torch.meshgrid(torch.arange(256.0), torch.arange(256.0), indexing="ij")
    img = (torch.sin(yy / 30) + torch.cos(xx / 45))[None, None]
    tx = torch.tensor([[100.0, 140.0]])
    back = pf.to_cartesian(pf.to_polar(img, tx), tx)
    err = (back - img).abs()[0, 0]
    assert float(err.mean()) < 0.01 and float(err.max()) < 0.1


def test_scan_is_running_max() -> None:
    """Training (cummax) and export (log-step) scans give the same output."""
    scan = RayScan(16, g=8)
    x = torch.randn(2, 16, 12, 64)
    ref = scan(x)
    scan.export_mode = True
    assert torch.allclose(scan(x), ref, atol=1e-6)


def test_e4_starts_as_e1b_and_export_mode_matches() -> None:
    """Zero head: E4 = E1b prior at init; export_mode on/off gives the same output."""
    h, tx = bumpy()
    model = PathlossNet("e4", "58", RANGES, COEFFS, width=8).eval()
    torch.nn.init.normal_(model.net.head.weight, std=1e-3)  # make the branch matter
    with torch.no_grad():
        ref = model(h, tx)
        model.set_export_mode(True)
        assert torch.allclose(model(h, tx), ref, atol=1e-5)
    fresh = PathlossNet("e4", "58", RANGES, COEFFS, width=8).eval()
    prior = PriorModel(COEFFS)(h, tx, torch.full((2,), BAND_HZ["58"]))
    with torch.no_grad():
        assert torch.allclose(fresh(h, tx), prior, atol=1e-3)


def test_every_parameter_gets_a_gradient() -> None:
    """All E4 parameters (polar branch included) get a gradient."""
    h, tx = bumpy(1)
    model = PathlossNet("e4", "58", RANGES, COEFFS, width=8)
    torch.nn.init.normal_(model.net.head.weight, std=1e-2)
    model(h, tx).mean().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None or not p.grad.any()]
    assert not missing, missing


def test_e5a_has_the_same_parameters_as_e4() -> None:
    """The ablation keeps every layer; only the scan operation is removed."""
    e4 = PathlossNet("e4", "58", RANGES, COEFFS, width=8)
    e5a = PathlossNet("e5a", "58", RANGES, COEFFS, width=8)
    assert count_params(e4) == count_params(e5a)
    assert PathlossNet.from_config(e4.config()).config() == e4.config()


def test_e4_onnx_parity(tmp_path: Path) -> None:
    """Full E4 exports to one ONNX file (dynamo, opset 18) and matches PyTorch within 0.01 dB."""
    h, tx = bumpy(1)
    model = PathlossNet("e4", "58", RANGES, COEFFS, width=8).eval()
    torch.nn.init.normal_(model.net.head.weight, std=1e-2)
    model.set_export_mode(True)
    with torch.no_grad():
        ref = model(h, tx).numpy()
    path = tmp_path / "e4.onnx"
    torch.onnx.export(model, (h, tx), str(path), dynamo=True, opset_version=18,
                      input_names=["height", "tx"], output_names=["pl_db"], external_data=False)
    assert not list(tmp_path.glob("*.data"))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"height": h.numpy(), "tx": tx.numpy()})[0]
    assert np.abs(out - ref).max() <= 0.01
