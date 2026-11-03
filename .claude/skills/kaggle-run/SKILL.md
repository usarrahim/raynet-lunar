---
name: kaggle-run
description: Prepare a Kaggle GPU run for one experiment and give the owner exact click-by-click steps.
disable-model-invocation: true
argument-hint: "<run name, for example E4_58_s0>"
---
Prepare the Kaggle run "$ARGUMENTS".

1. Check that the config exists in `configs/` and that the CPU smoke test passes with it.
   Check that the code is committed and pushed to GitHub; note the commit hash.
2. Update `notebooks/kaggle_runner.ipynb` (template in `docs/06_COMPUTE.md`) with this run name,
   config, seed and commit hash.
3. Estimate the GPU hours from `configs/base.yaml` and earlier `summary.json` files. Show the
   remaining budget from the GPU hour log in `PROGRESS.md`.
4. Give the owner numbered steps in Turkish: open the notebook on Kaggle, check
   Accelerator = GPU and Internet = On, run the smoke cell, then
   "Save Version → Save & Run All (Commit)", where to find the Output, which files to download,
   and where to put them (`runs/$ARGUMENTS/`).
5. Add a "running" line to `PROGRESS.md` (date, run name, commit, estimated hours).
