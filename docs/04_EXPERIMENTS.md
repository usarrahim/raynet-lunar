# 04 · Experiments

## Order (5.8 GHz first: it has the largest error and is the main test of H1)
| ID | What | Band | Seeds | Needs GPU | Answers |
|---|---|---|---|---|---|
| E0 | RadioUNet, official code, our budget | 58, then 415 | 1 | yes | Our reference point |
| E1 | Prior FSPL + J | both | - | no | How good is physics alone? |
| E1b | Calibrated prior (5 coefficients) | both | - | no | Base for E3/E4 |
| E2 | U-Net + physics channels | 58 | 1 | yes | Do extra inputs help? |
| E3 | E2 + residual on E1b | 58 (then 415) | 3 | yes | Does the prior help? Control for H1 |
| E4 | RayNet | 58 (then 415) | 3 | yes | Main idea (H1, H2) |
| E5a | E4 without scan | 58 | 1 | yes | Is the scan the reason? |
| E5b | Parameter-matched Cartesian | 58 | 1 | yes | Is it only more parameters? |
| E5c | One conditional E4 (both bands) | both | 1 | yes | Optional |
| E6 | Test-time augmentation on best | both | - | inference only | Extra gain |

For H3 and the combined score, E0, E3 and E4 also need a 415 MHz run (1 seed is acceptable if
the GPU budget is short; say so in the report).

## Equal budget rule
All trained models (E0, E2-E5) use the same `configs/base.yaml` budget: same number of epochs
(E0 split 75/25 over its two stages), same batch size, same augmentation, same dev split.
Only the model changes. If a run crashes and resumes, it must reach the same number of epochs.

## Seeds
Seeds 0, 1, 2. Seed 0 for single-seed runs. The seed controls weight init, data order and
augmentation. Log it.

## Run names and folders
`<ID>_<band>_s<seed>`, for example `E4_58_s1`. Each run folder `runs/<name>/` contains:
- `config.yaml` (full resolved config), `git_commit.txt`
- `metrics.csv`: epoch, train_loss, dev_rmse_db, lr, time_s (dev is only monitored, no early stopping)
- `last.pt` (the evaluated model) and `best_dev.pt` (not in git; they stay on Drive)
- `summary.json`: params, best epoch, dev RMSE, train time, GPU type, peak memory
- After evaluation: `val_per_map.csv` (one row per map: sample_id, terrain_id, n_valid,
  rmse_all, rmse_los_core, rmse_nlos_core, rmse_border, sse per region, coverage accuracy)
  and `val_summary.json`.

## GPU budget plan (estimate, adjust after P2.2 speed test)
| Block | Runs | Hours (est.) |
|---|---|---|
| Debug and smoke tests | - | 4 |
| E0 58 + 415 | 2 | 4 |
| E2 58 | 1 | 1.5 |
| E3 58 × 3 seeds, E4 58 × 3 seeds | 6 | 10 |
| E5a, E5b | 2 | 3.5 |
| E3, E4 at 415 | 2 | 3.5 |
| Evaluation, timing, TTA | - | 2 |
| **Total** | | **≈ 29 h** |
Colab units are the main budget; the free Kaggle GPU quota is the reserve. Track the real hours in `PROGRESS.md`.

## Results table (fill only from `val_summary.json` files)
| ID | 415 MHz RMSE (dB) | 5.8 GHz RMSE (dB) | Combined | NLOS 5.8 | Border 5.8 | Params | ms/map |
|---|---|---|---|---|---|---|---|
| E0 | TBD | TBD | TBD | TBD | TBD | 13.27M | TBD |
| E1 | TBD | TBD | TBD | TBD | TBD | 0 | TBD |
| E1b | TBD | TBD | TBD | TBD | TBD | 10 | TBD |
| E2 | - | TBD | - | TBD | TBD | TBD | TBD |
| E3 | TBD | TBD ± TBD | TBD | TBD | TBD | TBD | TBD |
| E4 | TBD | TBD ± TBD | TBD | TBD | TBD | TBD | TBD |
| E5a | - | TBD | - | TBD | TBD | TBD | TBD |
| E5b | - | TBD | - | TBD | TBD | TBD | TBD |
| E6 | TBD | TBD | TBD | TBD | TBD | = best | TBD |
Reference only (hidden test set, 450+150 epochs, not comparable): RadioUNet 4.66 / 6.28 / 5.47 dB.

## What to do if results are surprising
- E3 worse than E2: check the prior normalization and the zero-init head.
- E4 not better than E3: check the polar mapping with a unit test (round trip Cartesian → polar
  → Cartesian of a smooth image should have small error), then look at E5a vs E4.
- Big gap between dev and val: check the dev split is by terrain, not by sample.
