# 03 · Models (E0-E6)

All models take `(height, tx, band)` and return pathloss in dB, shape (B, 1, 256, 256).
Internally they work in normalized units: `y_norm = (PL − pl_min) / (pl_max − pl_min)`
(ranges from `metadata.json`). Physics features are computed in **float32, outside autocast**
(atan2 and log10 are not safe in fp16).

## E0: RadioUNet baseline (official code)
- Use the official repository (`train.py`, RadioWNet, 13.27M parameters, two stages).
- Same total budget as our models (`configs/base.yaml`): stage 1 = 75% of the epochs,
  stage 2 = 25% (the official 450 : 150 ratio). Masked loss. Official augmentation.
- The official script selects "best" on val. We do not use that: stage 2 starts from stage 1's
  `_last.pt` (`--init-from <path>`), and we evaluate stage 2's `_last.pt`.
- Run on Colab by cloning the official repo inside the notebook (start it through
  `scripts/run_official.py`, docs/06_COMPUTE.md). Do not copy its code into
  our package. Evaluate its checkpoint with OUR evaluation code (same metrics and regions).

## E1 / E1b: physics prior (no training)
- E1: `PL = FSPL + J` (docs/02_PHYSICS.md).
- E1b: calibrated prior with 5 least-squares coefficients per band (fit on dev split).

## Shared backbone: `SmallUNet`
- 4 down levels (256 → 16), channels `[w, 2w, 4w, 8w]`, bottleneck `8w`.
- Each level: 2 × (Conv3×3 → GroupNorm(8) → SiLU). Down: MaxPool 2×2. Up: bilinear ×2 + Conv3×3.
  Skip connections by concatenation. Head: Conv1×1 → 1 channel.
- `w` is chosen in task P2.2 from a speed test (24 or 32). Print and log the parameter count.

## E2: U-Net + physics channels
- Input: the 9 channels of docs/02_PHYSICS.md section 8. Output: `y_norm` directly.

## E3: U-Net + physics channels + residual
- Same input as E2. Output Δ; prediction `y_norm = prior_norm(E1b) + Δ`.
- Initialise the last layer to zero, so training starts from the prior.
- E3 is the main control for H1: everything is the same as E4 except the polar branch.

## E4: RayNet
```
Cartesian channels (9) ─────────────────────────────────────────────┐
                                                                    ├─ concat → SmallUNet → Δ → y = prior + Δ
height, tx ─ polar grid (T=360, R=256) ─ ray encoder + ray scans ─ back to Cartesian (C=32) ┘
```
1. **Polar input (7 channels, on the network grid T = 360 rays, R = 256 radii, Δr = 363/256 m):**
   terrain height relative to the TX antenna `(H − z_tx)/50`, terrain angle α (deg/30),
   horizon A (deg/30), LoS, shadow (deg/10), ν/5, inside-tile mask.
   Compute them with the physics module on this grid (same code, other T and R).
   Tensor layout: (B, C, T, R). The θ axis is "height", the r axis is "width".
2. **Ray encoder (C = 32):**
   - Stem: Conv2d(7 → C, kernel (1, 7), padding (0, 3)).
   - `RayBlock(d)`: Conv2d(C, C, kernel (1, 5), dilation (1, d)) along the ray → circular pad on
     θ (`F.pad(x, (0, 0, 1, 1), mode="circular")`) → Conv2d(C, C, kernel (3, 1)) across rays →
     GroupNorm(8) → SiLU, with a residual connection.
   - `RayScan`: `g = Conv1×1(C → 16)`; `S = cumulative max of g along r` (inclusive, from the TX
     outward; this is a learned "horizon"); `out = F + Conv1×1(concat[F, S, g − S] → C)`.
   - Order: Stem → RayBlock(1) → RayScan → RayBlock(2) → RayScan → RayBlock(4).
3. **Back to Cartesian:** the same `grid_sample` mapping as the physics module (wrap θ).
4. **Fusion:** concat with the 9 Cartesian channels (41 channels) → SmallUNet → Δ (zero-init head).
5. **Scan implementation:** in training, `torch.cummax(g, dim=-1).values` (fast, less memory).
   For ONNX export, the inclusive version of `scan_max_exclusive` (`max(exclusive, g)`).
   A switch `model.export_mode = True` selects it. Test: both modes give the same output
   (max difference < 1e-5).

## E5: ablations
- **E5a, no scan:** E4 with `S` replaced by `g` (same layers, same parameter count). Tests the scan.
- **E5b, parameter-matched Cartesian:** E3 with a wider U-Net so that its parameter count is
  within ±5% of E4. Tests "is it only more parameters?"
- **E5c (optional), one conditional model:** E4 trained on both bands with a band channel,
  compared with two band-specific E4 models.

## E6: test-time augmentation
- For the best model: run the 8 dihedral transforms of (height, TX), undo each transform on the
  output, average. Report separately (it costs 8× inference time).

## Training defaults (`configs/base.yaml`)
| Key | Value |
|---|---|
| optimizer | AdamW, lr 1e-3, weight decay 1e-4 |
| schedule | 1 epoch linear warm-up, then cosine to 1e-5 |
| batch size | 16 (reduce to 8 if GPU memory is short; then keep it for all runs) |
| precision | AMP fp16 autocast + GradScaler (physics features in fp32) |
| gradient clip | 1.0 |
| augmentation | 8 dihedral transforms, train only |
| loss | masked MSE on `y_norm` |
| epochs | set in P2.2 so that one 5.8 GHz run takes about 1.5 h on the chosen Colab GPU |
| checkpoint | `last.pt` is the model we evaluate (no early stopping); `best_dev.pt` only for curves |

## Parameter and speed log
Every model prints: parameter count, forward time for batch 1 on GPU (median of 100 runs after
10 warm-up runs), and peak GPU memory for a training step. Write them to `summary.json`.
