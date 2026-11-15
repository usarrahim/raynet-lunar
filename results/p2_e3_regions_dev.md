# E3 vs E1b vs P1.6: error by region and distance (dev, 5.8 GHz)

Made by `scripts/region_report.py` from: `results\eval\P16_58_s0_firstU\dev_summary.json`, `results\eval\E1b_58\dev_summary.json`, `results\eval\E3_58_s0\dev_summary.json`

Each cell: RMSE (dB) / share of valid pixels (%) / share of the squared error (%).

| Model | Split | RMSE all | los_core RMSE / px % / SSE % | border RMSE / px % / SSE % | nlos_core RMSE / px % / SSE % | nlos+border SSE % |
|---|---|---|---|---|---|---|
| P16_58_s0_firstU | dev | 14.58 | 8.42 / 82.0 / 27.4 | 23.85 / 7.6 / 20.4 | 32.73 / 10.4 / 52.2 | 72.6 |
| E1b | dev | 6.64 | 5.14 / 82.0 / 49.2 | 13.64 / 7.6 / 32.2 | 8.89 / 10.4 / 18.6 | 50.8 |
| E3_58_s0 | dev | 5.10 | 4.31 / 82.0 / 58.6 | 9.95 / 7.6 / 29.1 | 5.55 / 10.4 / 12.3 | 41.4 |

## RMSE (dB) by distance from the TX

| Model | 0-25m | 25-50m | 50-100m | 100-150m | 150-200m | 200-363m |
|---|---|---|---|---|---|---|
| P16_58_s0_firstU | 1.20 | 4.19 | 12.17 | 16.21 | 17.22 | 16.63 |
| E1b | 0.94 | 2.70 | 5.61 | 7.23 | 7.87 | 7.57 |
| E3_58_s0 | 0.91 | 2.24 | 4.37 | 5.51 | 5.95 | 5.90 |

## Gate G1

Gate G1 (P16_58_s0_firstU, dev): nlos_core + border hold 72.6% of the squared error (threshold 40%) -> **PASS**
