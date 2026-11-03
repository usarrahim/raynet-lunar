---
name: add-result
description: Read a finished run from runs/, check it, and update PROGRESS.md and the results table.
disable-model-invocation: true
argument-hint: "<run name>"
---
Add the results of run "$ARGUMENTS".

1. Read `runs/$ARGUMENTS/summary.json`, `metrics.csv`, and `val_summary.json` if it exists.
   If a file is missing, say which one and stop.
2. Sanity check: the loss goes down, no NaN, the number of epochs equals the budget, RMSE values
   are in a sensible range (about 1-30 dB).
3. Update the results table in `docs/04_EXPERIMENTS.md` using only these files (keep "TBD"
   elsewhere), and the GPU hour log in `PROGRESS.md`.
4. For 5.8 GHz runs of E3, E4 or E5, update the H1/H2 status in `PROGRESS.md`
   (say clearly if some seeds are still missing).
5. Save `results/figures/<run name>_curve.png` (train loss and dev RMSE per epoch).
6. Answer in Turkish: 3-5 bullets on what the result means compared with earlier runs, and the
   next step.
