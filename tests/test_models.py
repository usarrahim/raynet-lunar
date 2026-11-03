"""Tests for SmallUNet and the model wrapper."""

import torch

from raynet.models.prior import PriorModel
from raynet.models.unet import SmallUNet, count_params
from raynet.models.wrapper import CHANNELS, PathlossNet, Ranges
from raynet.physics import BAND_HZ

RANGES = Ranges(0.0, 496.0, 43.0, 228.0)
COEFFS = [15.9, 1.03, 2.3, -18.3, -9.9]


def bumpy(b: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Random smooth terrain batch and TX positions."""
    torch.manual_seed(0)
    h = torch.nn.functional.avg_pool2d(torch.rand(b, 1, 256, 256) * 400, 31, 1, 15)
    tx = torch.tensor([[40.0, 200.0], [128.0, 7.0]])[:b]
    return h, tx


def test_unet_shape_and_zero_head() -> None:
    """Output keeps the spatial size; zero head gives exactly 0."""
    net = SmallUNet(9, width=8, zero_head=True)
    y = net(torch.randn(2, 9, 256, 256))
    assert y.shape == (2, 1, 256, 256) and torch.all(y == 0)
    assert count_params(net) > 0


def test_e3_starts_as_the_e1b_prior() -> None:
    """With the zero-init head, E3 outputs exactly the E1b prior."""
    h, tx = bumpy()
    model = PathlossNet("e3", "58", RANGES, COEFFS, width=8).eval()
    prior = PriorModel(COEFFS)(h, tx, torch.full((2,), BAND_HZ["58"]))
    with torch.no_grad():
        assert torch.allclose(model(h, tx), prior, atol=1e-3)


def test_input_channels() -> None:
    """9 channels, finite, TX one-hot has exactly one pixel at the TX."""
    h, tx = bumpy()
    model = PathlossNet("e2", "58", RANGES, COEFFS, width=8)
    x, _ = model.inputs(h, tx)
    assert x.shape == (2, len(CHANNELS), 256, 256) and torch.isfinite(x).all()
    onehot = x[:, CHANNELS.index("tx_onehot")]
    assert onehot.sum(dim=(1, 2)).tolist() == [1.0, 1.0]
    assert onehot[0, 40, 200] == 1 and onehot[1, 128, 7] == 1
    log_d = x[:, CHANNELS.index("log_distance")]
    assert 0 <= float(log_d.min()) and float(log_d.max()) <= 1.01


def test_config_roundtrip_and_gradients() -> None:
    """Rebuild from config gives the same output; every parameter gets a gradient."""
    h, tx = bumpy()
    model = PathlossNet("e2", "415", Ranges(0, 496, 20, 193), COEFFS, width=8)
    copy = PathlossNet.from_config(model.config())
    copy.load_state_dict(model.state_dict())
    with torch.no_grad():
        assert torch.allclose(model(h, tx), copy(h, tx))
    model(h, tx).mean().backward()
    assert all(p.grad is not None for p in model.parameters())
