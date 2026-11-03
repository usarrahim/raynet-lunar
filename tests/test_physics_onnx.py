"""ONNX parity: the physics prior exported to ONNX matches PyTorch within 0.01 dB."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from raynet.physics import BAND_HZ, PhysicsFeatures


class PriorOnly(torch.nn.Module):
    """(height, tx, freq_hz) -> prior dB, the smallest model the website needs."""

    def __init__(self) -> None:
        """Wrap the physics features."""
        super().__init__()
        self.pf = PhysicsFeatures()

    def forward(self, height: torch.Tensor, tx: torch.Tensor, freq_hz: torch.Tensor) -> torch.Tensor:
        """Prior channel only."""
        return self.pf(height, tx, freq_hz)[:, 5:6]


def test_onnx_parity(tmp_path: Path) -> None:
    """Export with dynamo, opset 18, one file; ONNX Runtime matches within 0.01 dB."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:256, 0:256]
    hm = sum(rng.uniform(-15, 15) * np.exp(-((yy - rng.uniform(0, 256)) ** 2
                                            + (xx - rng.uniform(0, 256)) ** 2) / 800.0)
             for _ in range(10))
    height = torch.tensor(hm - hm.min(), dtype=torch.float32)[None, None]
    tx = torch.tensor([[70.0, 190.0]])
    freq = torch.tensor([BAND_HZ["58"]])
    model = PriorOnly().eval()
    with torch.no_grad():
        ref = model(height, tx, freq).numpy()

    path = tmp_path / "prior.onnx"
    torch.onnx.export(model, (height, tx, freq), str(path), dynamo=True, opset_version=18,
                      input_names=["height", "tx", "freq_hz"], output_names=["prior_db"],
                      external_data=False)
    assert not list(tmp_path.glob("*.data"))  # one file, no external weights
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"height": height.numpy(), "tx": tx.numpy(),
                          "freq_hz": freq.numpy()})[0]
    assert np.abs(out - ref).max() <= 0.01
