---
name: check
description: Run the project health check (tests, lint, CPU smoke run, ONNX parity, git hygiene) and report in Turkish.
disable-model-invocation: true
---
Run these and report the result of each one:

1. `pytest -q`
2. `ruff check src tests` (skip if ruff is not installed)
3. If `src/raynet/train.py` exists: CPU smoke test, 1 epoch, `--limit-batches 3`, on `data/mini`
   or the local val files, output to a temporary folder.
4. If `src/raynet/export_onnx.py` exists: export and run the ONNX parity test.
5. `git status`: make sure no data, checkpoints, `.npy` files or keys are staged.

Report in Turkish: what passed, what failed, the first error message, and a suggested fix.
If the fix is more than a few lines, ask before changing code.
