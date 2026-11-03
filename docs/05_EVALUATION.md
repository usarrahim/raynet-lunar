# 05 · Evaluation

One evaluation script for every model: `python -m raynet.evaluate`. It reads predictions in dB,
the targets and the masks, and writes `val_per_map.csv` + `val_summary.json`.
(E0 checkpoints from the official code are wrapped so they go through the same script.)

## 1. Main metric (same as the challenge)
- Masked RMSE in dB, **pooled over all valid pixels of all maps** of a split and band:
  `RMSE = sqrt( Σ_maps Σ_valid (pred − gt)² / Σ_maps n_valid )`.
  (The official `metrics.py` is pixel-weighted in the same way.)
- Combined score: `0.5 · RMSE_415 + 0.5 · RMSE_58`.
- Diagnostic only: all-pixel RMSE (includes the gap-filled pixels).

## 2. Regions (disjoint, inside the valid mask)
- Geometric LoS from the physics module (T = 720, Δr = 1 m).
- Boundary pixels: LoS differs from at least one 4-neighbour.
- **border** = pixels within 3 px (Euclidean) of a boundary pixel.
- **los_core** = LoS and not border. **nlos_core** = NLOS and not border.
- Distance bins (horizontal metres): [0,25), [25,50), [50,100), [100,150), [150,200), [200,363].
- For each region: pooled RMSE, mean error (bias, pred − gt), pixel share.

## 3. Hypothesis tests
- **H1:** E4 vs E3 at 5.8 GHz.
  - Pooled RMSE for each seed → mean ± std over 3 seeds. Effect = mean(E3) − mean(E4) ≥ 0.3 dB.
  - Per-map RMSE averaged over the 3 seeds → paired Wilcoxon signed-rank test over the 2,330
    val maps (`scipy.stats.wilcoxon`). p < 0.05.
  - Maps on the same terrain are not independent (51 TX share a terrain). Also run the test on
    terrain means (680 terrains) and a terrain-level bootstrap (1,000 resamples, 95% CI of the
    RMSE difference). Report all three.
- **H2:** squared-error reduction by region,
  `ΔSSE_region = SSE_E3,region − SSE_E4,region` (summed over val maps, mean over seeds).
  Share = `(ΔSSE_nlos_core + ΔSSE_border) / ΔSSE_total` ≥ 0.5.
- **H3:** combined RMSE of E4 ≤ combined RMSE of E0 − 0.5 dB on val, and inference < 500 ms/map.

## 4. Coverage accuracy
- Covered if PL ≤ 120 dB (RSS ≥ −100 dBm, since RSS = 20 − PL).
- Accuracy = share of valid pixels where prediction and ground truth agree on covered / not
  covered. Also report F1 for the "not covered" class (the dangerous mistake is "says covered
  but is not").
- Also plot accuracy for thresholds −110 … −70 dBm (PL 130 … 90 dB).

## 5. Speed
- GPU: batch 1, physics features included, 20 warm-up runs, then median of 200 maps,
  `torch.cuda.synchronize()` around the timer. Report GPU type, fp32 and fp16.
- CPU: same, batch 1, 4 threads.
- Browser: ONNX Runtime Web (WASM), median of 20 runs on the owner's laptop; report the CPU.
- Requirement: under 500 ms per map (dataset rule).

## 6. Figures for the report (save to `results/figures/`, PNG 200 dpi + the data as CSV)
1. Examples: height (hillshade), ground truth, E0, E3, E4, error maps (E3, E4) for 4 val maps:
   mostly LoS, strong shadow, TX near the edge, 5.8 GHz with large voids. Gray = invalid pixels.
2. Bar chart: RMSE by region (los_core, border, nlos_core) for E0, E1b, E3, E4.
3. RMSE vs distance bin (line chart).
4. Coverage accuracy vs threshold.
5. Training curves: dev RMSE vs epoch, E3 vs E4 (3 seeds, mean ± std band).
6. Per-map RMSE scatter E3 vs E4 (points under the diagonal: E4 is better).
Use one colour per model everywhere. Colour scale for PL maps: same limits for ground truth and
prediction in a figure.

## 7. Honesty checklist (before writing the report)
- [ ] All numbers come from `val_summary.json` files; the run names are listed in the report.
- [ ] Model selection used only the dev split.
- [ ] Organizer numbers are labelled "hidden test set, longer training".
- [ ] Failed or negative results are reported too.
- [ ] The diffraction-disabled fact is stated where the prior is discussed.
