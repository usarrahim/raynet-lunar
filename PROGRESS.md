# PROGRESS

## Status
- Current phase: Phase 0 (setup)
- Next task: P0.4 owner steps on Colab (Secrets, data zip to Drive), then the P1.6 training run
  (moved earlier, owner approved 2026-11-03). In parallel on the laptop: P1.1 loader.
- Last update: 2026-11-03

## Decisions so far
- 2026-10-29: Topic and plan fixed in HW1 (RayNet, lunar pathloss maps). The course project
  comes first; the ICASSP 2027 challenge entry is optional.
- 2026-11-03: Project kit prepared (CLAUDE.md, docs, skills). Changes after reading the
  official dataset documents:
  1. Diffraction is disabled in the dataset ray tracer → the knife-edge term J is a shadow-depth
     feature, not ground truth physics; E1 is expected to be weak in shadow.
  2. The organizer baseline numbers are hidden-test-set numbers after 450+150 epochs → we
     compare with our own budget-matched E0 on the frozen val split (H3 updated).
  3. No early stopping: fixed budget, evaluate the last checkpoint; dev split (5% of training
     terrains) for tuning and decisions.
- 2026-11-03: Website plan: static site + onnxruntime-web. The physics part was tested in a
  browser (WASM): about 70 ms per map, 0.0002 dB difference to Python. Preview page in
  `web/prototype/index.html`.

## In progress
- P0.4 Colab runner: `notebooks/colab_runner.ipynb` written (pinned to commit `1d2db99`).
  Waiting for the owner: Kaggle token in Colab Secrets, data zip to Drive, first run of the cells.
- P2.2 Training loop: code done and smoke-tested on the laptop. Waiting for the Colab speed
  test (notebook section 6) to fix `epochs` and `model.width` in `configs/base.yaml`
  (both PROVISIONAL now).
- P1.6 quick official baseline: ready in the notebook (section 7); waiting for the owner to run it.

## Done
- P1.1 Loader + dev split (2026-11-03): `src/raynet/data.py`, `splits/dev_terrains.txt`
  (20 of 392 multi + 286 of 5728 single terrains; 24414 train / 1306 dev / 2330 val maps per band).
- P1.2 Metrics (2026-11-03): `src/raynet/metrics.py`.
- P1.3 Physics (2026-11-03): `src/raynet/physics.py`, all tests of docs/02_PHYSICS.md §9 pass.
- P1.4 Evaluation (2026-11-03): `python -m raynet.evaluate`, writes `<split>_per_map.csv` and
  `<split>_summary.json`. Results table is made by `scripts/make_results_table.py`.
- P1.5 E1 and E1b (2026-11-03): coefficients in `results/e1b_coeffs.json` (fit on dev only);
  summaries in `results/eval/E1*_*/`. Table rows E1, E1b filled (see docs/04_EXPERIMENTS.md).
- P2.1 Model wrapper + SmallUNet (2026-11-03): `src/raynet/models/unet.py`, `wrapper.py`.
- P0.1 Repo skeleton (2026-11-03)
- P0.2 GitHub: https://github.com/usarrahim/raynet-lunar (public, branch `main`) (2026-11-03)
- P0.3 Data check (2026-11-03): the FULL train + val data is on the laptop (`data/LunarRM`, 9 GB).
  `check_data` passes. Numbers: `results/p0_check_data.txt`. Figure: `results/figures/p0_samples.png`.
  This also meets the P0.4 acceptance numbers (25,720 / 2,330 maps per band).

## Setup checklist
- [x] `setup/claude-skills/` copied to `.claude/skills/` (2026-11-03)
- [x] Colab MCP server set up in `.mcp.json` (uv 0.13.7, server starts) (2026-11-03)
- [x] New session started; skills and `colab-mcp` visible; Colab connection tested (2026-11-03:
  connection OK; GPU runtime gave A100-SXM4-40GB, Python 3.13.16, torch 2.11.0+cu130)
- [x] Laptop venv `.venv` with Python 3.11.9 and torch 2.11.0+cu128 (RTX 3050 Ti, CUDA works) (2026-11-03)

## Results so far
All TBD. Table: `docs/04_EXPERIMENTS.md`.

## GPU hour log
| Date | Platform | GPU | Run | Hours |
|---|---|---|---|---|

## Open questions
- Weekly Kaggle GPU quota of this account (see Kaggle settings).
- Rules on external data and pretrained weights are not stated by the organizers (our plan
  does not need them).

## Log (newest first: date · task · what changed · how to run · result)
- 2026-11-03 · P2.1 + P2.2 (code) · `SmallUNet` (4 levels, GroupNorm, SiLU, optional zero
  head), `PathlossNet` (9 channels of docs/02_PHYSICS.md §8 computed in fp32 outside autocast;
  normalization from metadata.json per band; E2 direct, E3 = E1b prior + Δ), `raynet.config`
  (base + override YAML), `configs/base.yaml` + E2/E3 configs, `raynet.train` (AdamW, warm-up +
  cosine per step, AMP, grad clip, masked MSE, seeded data order per epoch, `last.pt` every
  epoch with atomic save, `--resume auto` with a budget check, `metrics.csv`, `summary.json`),
  `raynet.evaluate --run` (LAST checkpoint). Physics grid buffers are no longer saved in
  checkpoints. · CPU and laptop-GPU smoke tests pass; E3 at step 0 equals E1b (tested). Laptop
  speed (RTX 3050 Ti, 60 steps, batch 16): w=24 2.33M params 48 maps/s 2.65 GB; w=32 4.14M
  params 38 maps/s 3.52 GB (this is NOT the Colab speed). · 52 tests pass.
- 2026-11-03 · P1.4 + P1.5 · `PriorModel` (E1, E1b), `raynet.calibrate` (least squares on
  all valid dev pixels, float64 normal equations), `raynet.evaluate`, `raynet.results_table`.
  Laptop GPU (RTX 3050 Ti): 2330 val maps in about 8 s. · `python -m raynet.calibrate
  --data-root data/LunarRM`, then `python -m raynet.evaluate --model e1b --band 58 --split val
  --data-root data/LunarRM`, then `python scripts/make_results_table.py` · Val results
  (from `results/eval/*/val_summary.json`): E1 7.45 / 13.11 dB, E1b 5.57 / 6.96 dB
  (415 / 5.8 GHz). E1b 5.8 GHz val: border + nlos_core are 18.7% of valid pixels but 51.9% of
  the squared error. E1b coefficients (`results/e1b_coeffs.json`): NLOS adds about 13 dB
  (415) and 18 dB (5.8) on top of FSPL, as expected with diffraction off. At 415 MHz almost no
  valid pixel is above 120 dB, so coverage F1 at that threshold is not informative; use the
  threshold sweep.
- 2026-11-03 · P1.3 · `PhysicsFeatures` from the verified prototype (same maths), split into
  `polar()` (polar maps incl. inside-tile mask, for RayNet later), `to_cartesian()` and
  `forward()`; `scan_max_inclusive` added for the learned scan. Tests: vectors, scan = cummax
  (R 256 and 363), flat, wall, LoS vs brute force on 20 real dev terrains (thresholds 99.0% /
  98.5% pass), rotation, ONNX parity (dynamo, opset 18, one file, ≤ 0.01 dB). ·
  `pytest -q` (physics tests take about 1.5 min) · all pass.
- 2026-11-03 · P1.2 · `MetricAccumulator`: pooled masked RMSE (divided by the valid pixel
  count), bias, per-map rows, all-pixel RMSE (diagnostic), regions los_core / border (3 px
  Euclidean around LoS changes) / nlos_core, distance bins, coverage accuracy and F1 of
  "not covered" at PL 120 dB (more thresholds possible). · `pytest -q` · 27 tests pass.
- 2026-11-03 · P1.1 · `LunarRMDataset` (one band, terrain filters, lazy memmaps that are not
  pickled to workers, 8 dihedral transforms with matching TX coordinates) and `make_datasets`
  (train minus dev / dev / val). · `pytest -q` · 19 tests pass; laptop reads about 466 maps/s
  with `num_workers=0`.
- 2026-11-03 · P0.4 / P1.1 / P1.6 prep · Colab is the main GPU platform: `docs/06_COMPUTE.md`
  rewritten, skill `kaggle-run` → `colab-run`, other docs updated. Official baseline checked
  (commit `fa3ef9b`): `--loss masked` (default) divides by `mask.sum()`; `--init-from` loads
  weights only, but `auto` prefers `_best.pt` (chosen on val) → our wrapper refuses `auto`.
  The official script trains on all train terrains and validates on `val` every epoch →
  `raynet.official_wrap` / `scripts/run_official.py` removes the dev terrains from training
  and uses them as the monitor set. Dev split made (`scripts/make_dev_split.py`). ·
  Local GPU smoke test of both official stages through the wrapper: 24414 train / 1306 dev
  samples, runs end to end. · 13 tests pass.
- 2026-11-03 · P0.3 · `src/raynet/check_data.py` (checks 1-7 of docs/01_DATA.md, chunked memmap
  reads, skips missing files) + `tests/test_check_data.py` (fake dataset). Data folder also has
  `manifest.json` / `subset_index.csv` (the Kaggle leaderboard is a 100-map val subset, marked
  "gameable": a format check only) and `val_terrains.txt`. No test inputs yet (December). ·
  `python -m raynet.check_data --root data/LunarRM` (about 1 min) · PASS; see
  `results/p0_check_data.txt` (for example 5.8 GHz train valid mean 89.62%, all voids = 145.0 dB).
- 2026-11-03 · P0.2 · Added remote `origin` and pushed `main`. Commits show only the owner as
  author (no AI co-author lines). · `git push` · Files visible on GitHub.
- 2026-11-03 · P0.1 · `pyproject.toml` (package `raynet`, src layout, extra `dev`),
  `src/raynet/__init__.py`, `tests/test_smoke.py`, empty folders `configs/ scripts/ notebooks/
  splits/ results/`, MIT `LICENSE`, `git init` + first commit. `KILAVUZ.pdf` and `PROMPTS_TR.md`
  are kept out of git (personal guides). · `.venv\Scripts\activate`, then
  `pip install -e ".[dev]"` and `pytest -q` · 1 test passed, ruff clean.
- 2026-11-03 · setup · Colab MCP connection tested in a new session; GPU runtime works.
- 2026-11-03 · setup · Copied skills to `.claude/skills/`. Installed uv. Added `.mcp.json`
  with the Colab MCP server (browser session mode). Owner chose Colab as the main GPU platform
  (docs/06_COMPUTE.md still says Kaggle; update it later). · Start a new session. · Connection test TBD.
