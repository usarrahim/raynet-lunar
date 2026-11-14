# P1.6 error by region and distance (dev, 5.8 GHz)

Made by `scripts/region_report.py` from: `results\eval\P16_58_s0_firstU\dev_summary.json`, `results\eval\E1b_58\dev_summary.json`

Each cell: RMSE (dB) / share of valid pixels (%) / share of the squared error (%).

| Model | Split | RMSE all | los_core RMSE / px % / SSE % | border RMSE / px % / SSE % | nlos_core RMSE / px % / SSE % | nlos+border SSE % |
|---|---|---|---|---|---|---|
| P16_58_s0_firstU | dev | 14.58 | 8.42 / 82.0 / 27.4 | 23.85 / 7.6 / 20.4 | 32.73 / 10.4 / 52.2 | 72.6 |
| E1b | dev | 6.64 | 5.14 / 82.0 / 49.2 | 13.64 / 7.6 / 32.2 | 8.89 / 10.4 / 18.6 | 50.8 |

## RMSE (dB) by distance from the TX

| Model | 0-25m | 25-50m | 50-100m | 100-150m | 150-200m | 200-363m |
|---|---|---|---|---|---|---|
| P16_58_s0_firstU | 1.20 | 4.19 | 12.17 | 16.21 | 17.22 | 16.63 |
| E1b | 0.94 | 2.70 | 5.61 | 7.23 | 7.87 | 7.57 |

## Gate G1

Gate G1 (P16_58_s0_firstU, dev): nlos_core + border hold 72.6% of the squared error (threshold 40%) -> **PASS**

## Notes (numbers from `results/eval/P16_58_s0_firstU/dev_summary.json`)
- P1.6 = official RadioWNet, stage 1 only, 10 epochs, trained without the dev terrains
  (`scripts/run_official.py`). Its own log on the same dev maps: 14.581 dB; our evaluation:
  14.579 dB (same model, two implementations agree).
- Bias in nlos_core is -31.09 dB and in border -15.40 dB: after this short training the network
  predicts far too little loss in shadow (it draws a smooth map without the sharp shadows).
- E1b (5 numbers, no training) is better in every region and distance bin here. This is a
  short-training result; it is not the final E0 (full budget, two stages).
