"""Tests for the config loader and the training loop (tiny fake data, CPU)."""

import csv
import itertools
import json
from pathlib import Path

import pytest
import yaml
from test_check_data import make_fake_root

from raynet import evaluate, train
from raynet.config import deep_merge, load_config


def test_deep_merge_and_base(tmp_path: Path) -> None:
    """Experiment files override only the keys they name."""
    (tmp_path / "base.yaml").write_text("a: 1\ntrain: {lr: 0.1, epochs: 5}\n")
    (tmp_path / "exp.yaml").write_text("base: base.yaml\ntrain: {epochs: 9}\n")
    cfg = load_config(tmp_path / "exp.yaml")
    assert cfg == {"a": 1, "train": {"lr": 0.1, "epochs": 9}}
    assert deep_merge({"x": {"y": 1}}, {"x": {"z": 2}}) == {"x": {"y": 1, "z": 2}}


def test_lr_schedule() -> None:
    """Warm-up rises to 1, cosine ends at lr_min / lr."""
    f = [train.lr_factor(s, 10, 100, 1e-3, 1e-5) for s in range(100)]
    assert f[0] == pytest.approx(0.1) and f[9] == pytest.approx(1.0)
    assert f[10] == pytest.approx(1.0) and f[-1] == pytest.approx(0.01, abs=1e-3)
    assert all(a >= b for a, b in itertools.pairwise(f[10:]))


def make_setup(tmp_path: Path) -> tuple[Path, Path]:
    """Fake data, a dev file, E1b coefficients and a small config."""
    root = make_fake_root(tmp_path)
    (tmp_path / "dev.txt").write_text("00001\n")
    coeffs = {b: {"coeffs": [10.0, 1.0, 1.0, -10.0, -1.0]} for b in ("415", "58")}
    (tmp_path / "e1b.json").write_text(json.dumps(coeffs))
    cfg = load_config(Path("configs/e3_residual_58.yaml"))
    cfg["model"]["width"] = 8
    cfg["train"].update(epochs=2, batch_size=1, num_workers=0, amp=False)
    cfg["data"] = {"dev_file": str(tmp_path / "dev.txt"), "e1b_coeffs": str(tmp_path / "e1b.json")}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return root, path


def run(root: Path, cfg: Path, out: Path, *extra: str) -> dict:
    """Call the training CLI on CPU."""
    return train.main(["--config", str(cfg), "--data-root", str(root), "--out", str(out),
                       "--device", "cpu", "--log-every", "0", *extra])


def test_train_writes_run_folder_and_resumes(tmp_path: Path) -> None:
    """2 epochs: run folder files exist; a second call with --resume auto does no extra epoch."""
    root, cfg = make_setup(tmp_path)
    out = tmp_path / "runs"
    s = run(root, cfg, out, "--epochs", "1")
    run_dir = out / "E3_58_s0"
    for name in ("config.yaml", "git_commit.txt", "metrics.csv", "summary.json", "last.pt"):
        assert (run_dir / name).exists(), name
    assert s["params"] > 0
    # resume needs the same budget: change epochs in the saved config like a real rerun would
    with pytest.raises(SystemExit):
        run(root, cfg, out, "--epochs", "2", "--resume", "auto")
    run(root, cfg, out, "--epochs", "1", "--resume", "auto")  # already done: no new epoch
    with open(run_dir / "metrics.csv", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 1


def test_evaluate_a_trained_run(tmp_path: Path) -> None:
    """evaluate --run reads last.pt and writes val files into the run folder."""
    root, cfg = make_setup(tmp_path)
    out = tmp_path / "runs"
    run(root, cfg, out, "--epochs", "1")
    s = evaluate.main(["--data-root", str(root), "--run", str(out / "E3_58_s0"),
                       "--split", "val", "--device", "cpu",
                       "--dev-file", str(tmp_path / "dev.txt")])
    assert s["model"] == "E3_58_s0" and s["band"] == "58"
    assert (out / "E3_58_s0" / "val_summary.json").exists()
