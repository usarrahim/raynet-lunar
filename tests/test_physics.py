"""Tests of docs/02_PHYSICS.md section 9 (vectors, scan, flat, wall, brute force, rotation)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from los_reference import los_bruteforce

from raynet.physics import (
    BAND_HZ,
    C_M_S,
    PhysicsFeatures,
    fresnel_nu,
    fspl_db,
    knife_edge_db,
    scan_max_exclusive,
    scan_max_inclusive,
)

VECTORS = json.loads(Path("docs/reference/test_vectors.json").read_text(encoding="utf-8"))
REAL_ROOT = Path("data/LunarRM")


def t(x: float) -> torch.Tensor:
    """Scalar float32 tensor."""
    return torch.tensor(x, dtype=torch.float32)


def features(height: np.ndarray, tr: float, tc: float, band: str = "58",
             n_theta: int = 720) -> torch.Tensor:
    """Run PhysicsFeatures on one tile; returns (6, S, S)."""
    pf = PhysicsFeatures(n_theta=n_theta)
    h = torch.as_tensor(height, dtype=torch.float32)[None, None]
    out = pf(h, torch.tensor([[tr, tc]], dtype=torch.float32), torch.tensor([BAND_HZ[band]]))
    return out[0]


def test_fspl_vectors() -> None:
    """FSPL within 0.01 dB of the test vectors."""
    for v in VECTORS["fspl_db"]:
        assert abs(float(fspl_db(t(v["d_m"]), t(v["f_hz"]))) - v["value"]) < 0.01


def test_band_frequencies_match_metadata() -> None:
    """Frequencies equal the vectors (and metadata.json if the data is here)."""
    for band, lam in VECTORS["wavelength_m"].items():
        assert abs(C_M_S / BAND_HZ[band] - lam) < 1e-5
    if (REAL_ROOT / "metadata.json").exists():
        meta = json.loads((REAL_ROOT / "metadata.json").read_text(encoding="utf-8"))
        assert {k: float(v) for k, v in meta["bands_hz"].items()} == BAND_HZ


def test_knife_edge_vectors() -> None:
    """J(nu) within 0.01 dB of the test vectors."""
    for v in VECTORS["knife_edge_J_db"]:
        assert abs(float(knife_edge_db(t(v["nu"]))) - v["value"]) < 0.01


def test_nu_vectors() -> None:
    """nu and J for the example edges."""
    for v in VECTORS["nu"]:
        lam = t(C_M_S / BAND_HZ[v["band"]])
        nu = fresnel_nu(t(v["h_m"]), t(v["d1_m"]), t(v["d2_m"]), lam)
        assert abs(float(nu) - v["nu"]) < 1e-3
        assert abs(float(knife_edge_db(nu)) - v["J_db"]) < 0.01


@pytest.mark.parametrize("r", [256, 363])
def test_scan_equals_cummax(r: int) -> None:
    """Exclusive scan equals shifted torch.cummax (values and indices); inclusive equals cummax."""
    torch.manual_seed(0)
    a = torch.randn(4, 30, r)
    m, i = scan_max_exclusive(a)
    ref_v, ref_i = torch.cummax(a, dim=-1)
    assert torch.equal(m[..., 1:], ref_v[..., :-1])
    assert torch.equal(i[..., 1:].long(), ref_i[..., :-1])
    assert torch.equal(scan_max_inclusive(a), ref_v)


def test_flat_terrain() -> None:
    """Flat terrain: LoS everywhere, J = 0, prior = FSPL."""
    f = features(np.full((256, 256), 7.0), 100, 80)
    assert torch.all(f[1] == 1)
    assert torch.all(f[4] == 0)
    assert torch.allclose(f[5], f[0])


def test_wall() -> None:
    """10 m wall at columns 150-152, TX at (128, 100): (128, 200) NLOS with J > 20 dB."""
    hm = np.zeros((256, 256))
    hm[:, 150:153] = 10.0
    f = features(hm, 128, 100)
    assert f[1, 128, 200] == 0 and f[4, 128, 200] > 20
    assert f[1, 128, 130] == 1


def terrain_cases(n: int) -> list[tuple[np.ndarray, int, int]]:
    """Real (dev split) terrains if the data is here, else smooth random terrains."""
    rng = np.random.default_rng(0)
    if (REAL_ROOT / "train_index.csv").exists():
        from raynet.data import make_datasets

        dev = make_datasets(REAL_ROOT, "58")["dev"]
        picks = rng.choice(len(dev), size=n, replace=False)
        out = []
        for p in picks:
            item = dev.get(int(p))
            r, c = (int(v) for v in item["tx"])
            out.append((item["height"][0].numpy().astype(np.float64), r, c))
        return out
    yy, xx = np.mgrid[0:256, 0:256]
    cases = []
    for _ in range(n):
        hm = np.zeros((256, 256))
        for _ in range(12):  # craters and hills as Gaussian bumps
            cy, cx = rng.uniform(0, 256, 2)
            hm += rng.uniform(-15, 15) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2)
                                                / (2 * rng.uniform(8, 40) ** 2))
        cases.append((hm - hm.min(), int(rng.integers(0, 256)), int(rng.integers(0, 256))))
    return cases


def test_los_vs_bruteforce() -> None:
    """LoS agrees with brute-force line tracing: mean >= 99.0%, worst >= 98.5% (20 cases)."""
    agree = []
    for hm, r, c in terrain_cases(20):
        los = features(hm, r, c)[1].numpy()
        agree.append(float((los == los_bruteforce(hm, r, c)).mean()))
    assert np.mean(agree) >= 0.990, agree
    assert min(agree) >= 0.985, agree


def test_rotation() -> None:
    """Rotating height and TX by 90 degrees rotates the LoS map (agreement >= 99%)."""
    hm, r, c = terrain_cases(1)[0]
    los = features(hm, r, c)[1].numpy()
    los_rot = features(np.rot90(hm).copy(), 255 - c, r)[1].numpy()  # np.rot90: (r,c)->(255-c,r)
    assert (np.rot90(los) == los_rot).mean() >= 0.99
