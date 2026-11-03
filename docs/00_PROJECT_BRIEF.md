# 00 · Project brief

This file is the short version of the approved HW1 proposal
(`docs/reference/HW1_proposal_EN.pdf`), plus facts we learned later from the official dataset
documents. If this file and the HW1 PDF disagree, this file wins. Section 6 explains why.

## 1. One sentence
RayNet predicts lunar radio coverage (pathloss) maps in milliseconds. It looks at the terrain
from the transmitter's point of view, scans every ray for the highest obstacle, and learns a
correction on top of a simple physics estimate.

## 2. Problem
- Future lunar networks (NASA LunaNet, Artemis, the Nokia 4G/LTE network on IM-2) need fast
  coverage planning. Craters, crater rims and hills block the signal.
- A pathloss map gives, for every pixel, how many dB the signal loses from the transmitter.
  Received power: RSS_dBm = 20 - PL_dB (20 dBm transmitter).
- Ray tracing (Sionna RT) gives accurate maps but is slow. It must run again for every new
  transmitter position.
- Neural networks are fast. The reference model RadioUNet treats the map like a normal image.
  Its official errors are 4.66 dB (415 MHz) and 6.28 dB (5.8 GHz). The hardest parts are the
  sharp shadows behind crater rims.

## 3. Our idea: three parts
1. **Physics prior (no training).** Free-space pathloss (FSPL) plus a single knife-edge term
   from the strongest obstacle on the TX→pixel profile.
2. **Transmitter-centred polar view + horizon scan.** Resample the map to an (angle, radius)
   grid around the TX, so each row is one ray. Along each ray, keep the running maximum of the
   elevation angle (the "horizon"). A point below the horizon is in shadow. A learnable version
   of this scan is inside the network.
3. **Learned residual.** Map back to the normal (Cartesian) grid. A small U-Net predicts a
   correction Δ. Output: PL = prior + Δ. Loss: masked MSE on valid pixels.

## 4. Hypotheses
- **H1 (main):** With the same backbone and the same training budget, RayNet (E4) lowers the
  5.8 GHz RMSE by at least 0.3 dB compared with E3 (U-Net with the same physics inputs and
  residual, no polar branch). Mean of 3 seeds; paired Wilcoxon test over validation maps, p < 0.05.
- **H2 (where):** At least half of that improvement (in squared error) comes from NLOS and
  shadow-border pixels (see `docs/05_EVALUATION.md` for exact regions).
- **H3 (practical):** RayNet beats our own budget-matched RadioUNet (E0) by at least 0.5 dB
  combined RMSE (50/50 of the two bands) on the frozen validation split. Inference must stay
  under 500 ms per map.
  - The HW1 version of H3 compared with the organizers' 5.47 dB. That number is from the hidden
    test set after 450+150 epochs. It can only be checked through the optional December
    challenge submission. For the course we use the version above.

## 5. Expected outputs
1. Open code on GitHub (MIT), with tests and configs.
2. Results table E0-E6 (val split, mean ± std over seeds).
3. Region analysis: LoS / NLOS / border / distance bins, and coverage accuracy at -100 dBm.
4. Demo website: pick a terrain, click to place the TX, see the predicted coverage map.
5. Final report and slides.

## 6. Facts we learned after HW1 (from official docs, Nov 2026)
Source: official dataset page and challenge repository. Summary: `docs/reference/official_dataset_notes.md`.
1. **Diffraction is disabled in the ray tracer.** Line-of-sight, specular reflection, diffuse
   scattering and refraction are modelled; diffraction is not. So in shadow, power arrives only
   through reflections and scattering, and many deep-shadow pixels have no rays at all (they are
   the invalid/void pixels).
   - Consequence: the knife-edge term J(ν) is a useful geometric feature (how deep the shadow
     is), not the true loss. E1 (prior only) is expected to be weak in shadow. We report this.
   - Consequence: shadow geometry (the horizon scan) matters even more, because shadow edges in
     the ground truth are sharp.
2. **Baseline numbers are test-set numbers after long training** (RadioWNet, 13.27M parameters,
   450 epochs stage 1 + 150 epochs stage 2). We cannot afford that. We compare models under one
   equal budget. Our E0 is the official code trained with that budget.
3. **Terrains are synthetic.** They come from the open LunarTerrainGenerator (MIT licence),
   which is tuned to real LRO data. We can generate new in-distribution terrains for the demo.
4. **Licence.** Dataset: CC BY 4.0 (attribution required). Challenge code: MIT.
5. **Data size.** About 10 GB in total; `train_pl415.npy` alone is 3.4 GB. Always use memory maps.
6. **Voids.** At 5.8 GHz about 11.4% of pixels are invalid (145.0 dB placeholder). Training on
   all pixels makes the ranked metric worse, so the loss must be masked.
7. **No early stopping (change from HW1).** The official code picks its "best" checkpoint on
   val. To keep val clean, every model is trained for the same fixed budget and we evaluate the
   last checkpoint. The dev split (5% of training terrains) is for tuning and decision gates.

## 7. What is borrowed and what is ours
- Borrowed: U-Net and RadioUNet, physics priors and residual learning (Earth radio map papers),
  polar transforms (vision), the dataset and the terrain generator.
- Ours: the ray-wise horizon scan on a transmitter-centred polar grid inside a radio map
  network, the test of whether it helps NLOS and border pixels on lunar terrain (H1, H2), the
  region-based error and coverage analysis, and the open demo.

## 8. Main risks and fallbacks
| Risk | Fallback |
|---|---|
| Errors are not mainly in shadows (weeks 1-2 analysis) | Change the method focus (for example near-TX region or reflection features) and report the finding |
| GPU quota runs out | 5.8 GHz only for the main comparison; fewer epochs; 1 seed for side experiments |
| Polar distortion near the TX | The Cartesian U-Net still sees Cartesian inputs; add a distance channel |
| ONNX export fails for the website | Host the PyTorch model with Gradio on Hugging Face Spaces (`docs/07_WEBSITE.md`) |
| RayNet does not beat E3 | Still a valid result: report it honestly (the "failure" answer in HW1) |
