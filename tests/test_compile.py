"""torch.compile gives the same E4 output as eager mode (needs a working compiler, else skipped)."""

import pytest
import torch

from raynet.models.wrapper import PathlossNet, Ranges


def compiler_works() -> bool:
    """True if torch.compile can build a tiny function on this machine (Triton on Linux)."""
    try:
        return bool(torch.compile(lambda x: x + 1)(torch.ones(1)) == 2)
    except Exception:  # noqa: BLE001 -- any compiler failure means "not available here"
        return False


@pytest.mark.skipif(not compiler_works(), reason="torch.compile not available here (Windows)")
def test_compiled_forward_matches_eager() -> None:
    """Same weights, same input: compiled and eager E4 agree within 1e-3 (normalized units)."""
    torch.manual_seed(0)
    h = torch.nn.functional.avg_pool2d(torch.rand(1, 1, 256, 256) * 400, 31, 1, 15)
    tx = torch.tensor([[40.0, 200.0]])
    model = PathlossNet("e4", "58", Ranges(0, 496, 43, 228), [15.9, 1.03, 2.3, -18.3, -9.9],
                        width=8).eval()
    torch.nn.init.normal_(model.net.head.weight, std=1e-2)
    with torch.no_grad():
        ref = model.forward_norm(h, tx)
        out = torch.compile(model.forward_norm)(h, tx)
    assert torch.allclose(out, ref, atol=1e-3)
