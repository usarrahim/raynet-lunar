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
- 2026-11-14: Contribution discussion. Owner approved two additions:
  1. Reflection-aware features (task P3.7, experiment E7): diffraction is off, so signal in
     shadow comes from reflections, which a ray-wise scan from the TX cannot see directly.
     Design after the E3 error analysis.
  2. Real lunar terrain (LOLA/LRO) in the demo website (task P4.5b), shown as a planning
     example without ground truth.
  Data-efficiency curve (train on 10/25/100% of the data) was proposed; not decided yet.
- 2026-11-14: E5b dropped (owner approved). At width 24, E3 has 2,329,225 parameters and E4
  2,367,913 (+1.7%), so E3 is already the parameter-matched control. Its GPU time goes to E7.

## In progress
- P0.4 Colab runner: `notebooks/colab_runner.ipynb` ready (uses `main`; runs log the exact
  commit). 2026-11-14: Kaggle Secrets set; data downloaded in Colab from Kaggle (zip 5.63 GB,
  under 1 minute) and saved on Drive as `MyDrive/raynet/data/LunarRM.zip` (5,628,361,638
  bytes; 19 files, same names and sizes as the laptop copy; flat layout). CPU runtime, no GPU.
  Next: GPU runtime → sections 1-4 → 5 (speed test) → 6 (P1.6).
- E4_58_s0 (P3.2): RUNNING on Colab since 2026-11-15 08:59 UTC (T4), 16 epochs,
  `MyDrive/raynet/runs/E4_58_s0/`; then evaluation on dev and val (same job). Speed on the T4:
  500 steps in about 4.8 min (≈ 1.7 steps/s ≈ 28 maps/s, E3: ≈ 73 maps/s), so about 15-16 min
  per epoch and about 4.2 h for the run (estimate). Peak GPU memory 5.81 GB.
  If the runtime stops: new GPU session → setup → `python -m raynet.train --config
  configs/e4_raynet_58.yaml --data-root /content/data/LunarRM --seed 0 --out
  /content/drive/MyDrive/raynet/runs --resume auto`, then evaluate dev and val.

## Done
- P2.5 E3 seed 0 at 5.8 GHz (2026-11-15): val RMSE 5.30 dB (results/eval/E3_58_s0/).
- P1.1 Loader + dev split (2026-11-03): `src/raynet/data.py`, `splits/dev_terrains.txt`
  (20 of 392 multi + 286 of 5728 single terrains; 24414 train / 1306 dev / 2330 val maps per band).
- P1.2 Metrics (2026-11-03): `src/raynet/metrics.py`.
- P1.3 Physics (2026-11-03): `src/raynet/physics.py`, all tests of docs/02_PHYSICS.md §9 pass.
- P1.4 Evaluation (2026-11-03): `python -m raynet.evaluate`, writes `<split>_per_map.csv` and
  `<split>_summary.json`. Results table is made by `scripts/make_results_table.py`.
- P1.5 E1 and E1b (2026-11-03): coefficients in `results/e1b_coeffs.json` (fit on dev only);
  summaries in `results/eval/E1*_*/`. Table rows E1, E1b filled (see docs/04_EXPERIMENTS.md).
- P2.1 Model wrapper + SmallUNet (2026-11-03): `src/raynet/models/unet.py`, `wrapper.py`.
- P1.6 Quick baseline + error map (2026-11-14): `results/p1_error_regions.md`. Gate G1 PASS
  (nlos_core + border = 72.6% of the squared error of the baseline on dev).
- P3.1 RayNet module (2026-11-14): `src/raynet/models/raynet.py` (polar branch, ray encoder,
  learned scan), kinds `e4` / `e5a` in the wrapper, configs `e4_raynet_58/415`, `e5a_noscan_58`.
- P2.2 Training loop + budget (2026-11-14): `src/raynet/train.py`; `configs/base.yaml` fixed:
  width 24, 16 epochs (speed test on Colab T4, see the log below).
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
| 2026-11-14 | Colab | T4 | setup + data check + P2.2 speed test (19:33-19:42 UTC) | 0.15 |
| 2026-11-14 | Colab | T4 | P1.6 official firstU, 10 epochs (19:42-20:43 UTC) + eval | 1.0 |
| 2026-11-14 | Colab | T4 | E3_58_s0 epochs 0-10 (20:44-21:48 UTC, runtime then stopped) | 1.1 |
| 2026-11-15 | Colab | T4 | setup + E3_58_s0 epochs 11-15 + eval (07:53-08:27 UTC) | 0.6 |
| 2026-11-15 | Colab | T4 | E4_58_s0, 16 epochs + eval (from 08:55 UTC) | TBD |

## Open questions
- Weekly Kaggle GPU quota of this account (see Kaggle settings).
- Rules on external data and pretrained weights are not stated by the organizers (our plan
  does not need them).

## Log (newest first: date · task · what changed · how to run · result)
- 2026-11-15 · P2.5 · E3_58_s0 (E3, 5.8 GHz, seed 0, width 24, 16 epochs, T4; epochs 0-10 on
  2026-11-14, runtime stopped, resumed at epoch 11 with `--resume auto`). Run logs copied from
  Drive to `results/runs/E3_58_s0/` (config, metrics, summary, commit `0cac9ab`), evaluation of
  the LAST checkpoint to `results/eval/E3_58_s0/`. Val: RMSE 5.295 dB (E1b 6.959, so 1.66 dB
  better), regions los_core 4.52 / border 10.15 / nlos_core 5.45 dB, bias -0.04 dB, coverage
  accuracy at 120 dB 97.3% (F1 not covered 0.910). Dev: 5.096 dB. Dev region report with P1.6
  and E1b: `results/p2_e3_regions_dev.md` (E3 better than E1b in every region and distance bin;
  border is now the largest shadow-related error: 9.95 dB RMSE, 29.1% of the SSE on dev).
  Dev curve: 5.88 dB (epoch 0) → 5.10 dB (epoch 15), still slowly improving at the end.
- 2026-11-14 · P1.6 · Official RadioWNet stage 1, 5.8 GHz, 10 epochs, seed 0, dev terrains out
  (Colab T4, about 6 min per epoch, 19:42-20:43 UTC). Its own dev log stays at about 15.3 dB for
  epochs 0-6 and ends at 14.581 dB (epoch 9). Our evaluation of `_last.pt` on dev: RMSE 14.579 dB,
  regions los_core 8.42 / border 23.85 / nlos_core 32.73 dB, bias in nlos_core -31.09 dB.
  Gate G1 PASS: nlos_core + border hold 72.6% of the squared error (threshold 40%).
  Report: `results/p1_error_regions.md`. · `python -m raynet.evaluate --official ...` then
  `python scripts/region_report.py ...` · The short-trained CNN misses the shadows; E1b
  (no training) is better in every region on dev.
- 2026-11-14 · P3.1 · `PolarBranch`: 7 polar inputs on T = 360, R = 256 (dr = 363/256 m,
  `PhysicsFeatures(n_r=...)`), stem + RayBlock(1) + RayScan + RayBlock(2) + RayScan +
  RayBlock(4), C = 32, back to Cartesian, concat with the 9 channels (41) → SmallUNet → Δ on
  E1b. `export_mode` switches cummax → `scan_max_inclusive`. `physics.to_polar` added (generic
  polar sampling). Tests: shapes, round trip, scan equality, export_mode equality, E4 = E1b at
  init, all parameters get gradients, E5a has the same parameters, full E4 ONNX export
  (dynamo, opset 18, one file) parity ≤ 0.01 dB. · `pytest -q` · 63 tests pass.
  Parameters (w = 24): E3 2,329,225, E4 2,367,913 (+1.7%). Laptop RTX 3050 Ti, batch 16:
  E3 41 maps/s, 2.65 GB; E4 peak 5.81 GB (more than the 4 GB card, so it spills to shared
  memory and runs at about 1 map/s; not usable on the laptop, fine on a 15 GB T4). E4 speed on
  the T4: TBD.
- 2026-11-14 · P2.2 speed test · Colab, Tesla T4 (15 GB), 2 CPUs, torch 2.14.1+cu130, commit
  `3c23f7e`. E3 at 5.8 GHz, 300 steps, batch 16, AMP, 2 workers: width 24 (2,329,225 params)
  76 maps/s, peak 2.65 GB, dev RMSE after 300 steps 6.045 dB (20 dev batches only); width 32
  (4,137,825 params) 61 maps/s, peak 3.52 GB, 6.117 dB. Decision: width 24, 16 epochs
  (24414 / 76 ≈ 5.4 min per epoch plus the dev pass → about 1.5 h per E3 run at 5.8 GHz).
  Reason: more passes over the data for the same GPU time; a wider model is tested anyway in
  E5b. Colab setup times: Drive zip → local 45 s, unzip about 2 min, full `check_data` PASS.
  P1.6 official firstU: about 2.9 steps/s (≈ 9 min per epoch) on the same T4.
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
