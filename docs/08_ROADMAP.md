# 08 · Roadmap (do the tasks in this order)

Rules: one task at a time; tick `[x]` when the acceptance check passes; write the result in
`PROGRESS.md`. Tasks marked **[GPU]** use GPU hours: ask the owner first. Tasks marked
**[Owner]** need the owner to do something by hand (Claude Code explains the steps).
"Learn" lists what the owner should be able to explain afterwards.

Plan: about 8 weeks from 3 Nov 2026. Optional challenge deadline in December (check the
official Timeline page).

## Phase 0: Setup (week 1)
- [x] **P0.1 Repo skeleton.** `pyproject.toml` (package `raynet`, src layout, extra `dev` =
  pytest + ruff), `src/raynet/__init__.py`, `tests/test_smoke.py`, folders `configs/ scripts/
  notebooks/ splits/ results/`, `LICENSE` (MIT, Abdurrahim Yılmaz, 2026), `git init`, first commit.
  *Accept:* `pip install -e ".[dev]"` and `pytest -q` pass on Windows.
  *Learn:* what a package is, why tests.
- [x] **P0.2 GitHub [Owner].** Owner makes an empty PUBLIC repo `raynet-lunar`; Claude Code adds the
  remote and pushes. *Accept:* files visible on GitHub.
- [x] **P0.3 Local data [Owner].** Owner downloads the val files and `metadata.json` from the
  Kaggle Data tab into `data/LunarRM/` (list in `docs/01_DATA.md`). Claude Code writes
  `src/raynet/check_data.py` (checks 1-7 in 01_DATA.md; skip checks whose files are missing).
  *Accept:* clean report and `results/figures/p0_samples.png`.
  *Learn:* the file format, masks, why memory maps.
- [ ] **P0.4 Colab runner [Owner].** `notebooks/colab_runner.ipynb` (docs/06_COMPUTE.md). Owner
  adds the Kaggle token to Colab Secrets, puts the data zip on Drive once, and runs `check_data`
  on Colab. *Accept:* 25,720 / 2,330 maps per band and valid fractions as in 01_DATA.md
  (already true on the laptop, see P0.3).

## Phase 1: Data, metrics, physics, first look at errors (weeks 1-2)
- [x] **P1.1 Loader + dev split.** `src/raynet/data.py`, `scripts/make_dev_split.py` →
  `splits/dev_terrains.txt` (5% of training terrains, seed 0; the full data is on the laptop, so the script runs
  there; commit the txt file).
  *Tests:* item shapes; augmentation keeps `height[tx]` unchanged; dev terrains do not appear in train.
  *Learn:* data leakage and why we split by terrain.
- [x] **P1.2 Metrics.** `src/raynet/metrics.py`: pooled masked RMSE, per-map RMSE, region masks,
  coverage accuracy. *Tests:* tiny hand-made arrays with known answers.
  *Learn:* pooled vs per-map RMSE; why masked.
- [x] **P1.3 Physics module.** `src/raynet/physics.py` from `docs/reference/physics_prototype.py`
  + all tests of docs/02_PHYSICS.md §9. *Accept:* all tests pass on CPU.
  *Learn:* FSPL, knife-edge, polar grid, horizon scan, why not `torch.cummax` in export.
- [ ] **P1.4 Evaluation script.** `src/raynet/evaluate.py` (+ `PriorModel` for E1). Writes
  `val_per_map.csv` and `val_summary.json` (docs/05_EVALUATION.md). *Accept:* runs locally on the
  val files with E1.
- [ ] **P1.5 E1 and E1b [laptop or Colab].** Evaluate the prior on dev and val
  for both bands; fit E1b on dev → `results/e1b_coeffs.json`.
  *Accept:* results table rows E1, E1b filled. *Learn:* why physics alone is weak in shadow here
  (diffraction is off in the data).
- [ ] **P1.6 Quick baseline + error map [GPU, about 2 h].** Official RadioUNet, 5.8 GHz, short
  run (10 epochs stage 1 only). Evaluate on dev with our regions.
  *Accept:* `results/p1_error_regions.md` with SSE share and RMSE per region and per distance bin.
  - **Gate G1:** if nlos_core + border hold less than 40% of the squared error, stop and discuss
    the method focus with the owner (this is the HW1 "weakest assumption" check).

## Phase 2: Physics inputs and residual (weeks 3-4)
- [ ] **P2.1 Model wrapper + SmallUNet.** Input channels (02_PHYSICS.md §8), `SmallUNet`,
  wrapper that returns dB. *Tests:* shapes; with a zero-init head the residual model outputs
  exactly the prior.
- [ ] **P2.2 Training loop + budget [GPU, about 1 h].** `src/raynet/train.py`: YAML config, AMP,
  masked loss, cosine LR, checkpoint + resume, `metrics.csv`, `summary.json`, flags
  `--epochs --limit-batches --seed --resume --out --run-name`. CPU smoke test first. Then on
  Colab measure samples/s for w = 24 and w = 32; choose w and N so that one 5.8 GHz run takes
  about 1.5 h; write `configs/base.yaml`. *Learn:* AMP, cosine LR, checkpoints.
- [ ] **P2.3 E0 full budget [GPU].** 5.8 GHz, then 415 MHz.
- [ ] **P2.4 E2 [GPU].** 5.8 GHz, seed 0.
- [ ] **P2.5 E3 [GPU].** 5.8 GHz, seed 0. Update the results table and region bar chart.
  *Learn:* residual learning; what changed between E0, E2 and E3 and why.
- [ ] **P2.6 (optional) Website W0.** Export the E1b prior to ONNX, build the site skeleton from
  `web/prototype/`, deploy (docs/07_WEBSITE.md).

## Phase 3: RayNet (weeks 5-6)
- [ ] **P3.1 RayNet module.** `src/raynet/models/raynet.py` (docs/03_MODEL.md E4).
  *Tests:* shapes; Cartesian → polar → Cartesian round trip of a smooth image has small error;
  `export_mode` on/off gives the same output; every parameter gets a gradient; full ONNX export
  + parity ≤ 0.01 dB. *Learn:* polar transform, circular padding, the learned scan.
- [ ] **P3.2 E4 seed 0 [GPU].** 5.8 GHz. Compare with E3 seed 0.
  - **Gate G2:** if E4 is more than 0.3 dB worse than E3, debug before spending more GPU hours.
- [ ] **P3.3 Seeds 1 and 2 for E3 and E4 [GPU].** 5.8 GHz.
- [ ] **P3.4 E5a and E5b [GPU].** 5.8 GHz.
- [ ] **P3.5 415 MHz: E3 and E4 [GPU].** Seed 0 (more seeds if budget allows). E5c optional.
- [ ] **P3.6 E6 test-time augmentation.** Best model, both bands.
  - **Gate G3:** if more than 75% of the GPU budget is used before Phase 4, drop E5c and extra seeds.

## Phase 4: Analysis, website, report (weeks 7-8)
- [ ] **P4.1 Hypothesis tests.** H1, H2, H3 (docs/05_EVALUATION.md §3) → `results/hypotheses.md`.
- [ ] **P4.2 Figures.** All figures of docs/05_EVALUATION.md §6.
- [ ] **P4.3 Speed.** GPU, CPU and browser timings.
- [ ] **P4.4 Export.** Best model to ONNX (contract in docs/07_WEBSITE.md) + `model_card.json`
  + parity test.
- [ ] **P4.5 Website W1.** Full demo, validation-tile comparison, results section, deploy.
- [ ] **P4.6 README.** Results table, demo link, how to reproduce, licences, AI-use statement.
- [ ] **P4.7 Report and slides.** docs/09_REPORT_GUIDE.md.
- [ ] **P4.8 (optional) Challenge submission [Owner].** Test inputs and deadline: see the official
  Timeline page (December). Final format: one `.npy` per test sample per band
  (`pl415_<sample_id>.npy`, `pl58_<sample_id>.npy`) + trained model + inference code; check with
  the official `score_submission.py --validate-only`. Under 500 ms per map.
