---
name: colab-run
description: Prepare a Colab GPU run for one experiment and give the owner exact click-by-click steps.
disable-model-invocation: true
argument-hint: "<run name, for example E4_58_s0>"
---
Prepare the Colab run "$ARGUMENTS".

1. Check that the config exists in `configs/` and that the CPU smoke test passes with it.
   Check that the code is committed and pushed to GitHub; note the commit hash.
2. Update `notebooks/colab_runner.ipynb` (template in `docs/06_COMPUTE.md`) with this run name,
   config, seed and commit hash. The run folder must be on Drive (`{DRIVE}/runs/`) and the run
   command must use `--resume auto`, so a disconnect loses at most one epoch.
3. Estimate the GPU hours from `configs/base.yaml` and earlier `summary.json` files. Show the
   GPU hours used so far from the log in `PROGRESS.md`. Ask the owner before any GPU use.
4. Give the owner numbered steps in Turkish: open the notebook in Colab (File → Open notebook →
   GitHub), Runtime → Change runtime type → GPU (which GPU and why), run the cells in order,
   check the smoke cell output, start the real run, keep the tab open. After the run: which
   files to copy from Drive (no `*.pt`) to `runs/$ARGUMENTS/` on the laptop, and to disconnect
   the runtime.
5. If the colab-mcp tools are connected, offer to run the cells yourself instead (the owner
   still chooses the GPU in the Colab menu).
6. Add a "running" line to `PROGRESS.md` (date, run name, commit, estimated hours).
