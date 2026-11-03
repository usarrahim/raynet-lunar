# PROGRESS

## Status
- Current phase: Phase 0 (setup)
- Next task: P0.4 (runner notebook; the data check part is already done on the laptop)
- Last update: 2026-11-03

## Decisions so far
- 2026-10-29: Topic and plan fixed in HW1 (RayNet, lunar pathloss maps). The course project
  comes first; the ICASSP 2027 challenge entry is optional.
- 2026-11-03: Project kit prepared (CLAUDE.md, docs, skills). Changes after reading the
  official dataset documents:
  1. Diffraction is disabled in the dataset ray tracer → the knife-edge term J is a shadow-depth
     feature, not ground truth physics; E1 is expected to be weak in shadow.
  2. The organizer baseline numbers are hidden-test-set numbers after 450+150 epochs → we
     compare with our own budget-matched E0 on the frozen val split (H3 updated).
  3. No early stopping: fixed budget, evaluate the last checkpoint; dev split (5% of training
     terrains) for tuning and decisions.
- 2026-11-03: Website plan: static site + onnxruntime-web. The physics part was tested in a
  browser (WASM): about 70 ms per map, 0.0002 dB difference to Python. Preview page in
  `web/prototype/index.html`.

## Done
- P0.1 Repo skeleton (2026-11-03)
- P0.2 GitHub: https://github.com/usarrahim/raynet-lunar (public, branch `main`) (2026-11-03)
- P0.3 Data check (2026-11-03): the FULL train + val data is on the laptop (`data/LunarRM`, 9 GB).
  `check_data` passes. Numbers: `results/p0_check_data.txt`. Figure: `results/figures/p0_samples.png`.
  This also meets the P0.4 acceptance numbers (25,720 / 2,330 maps per band).

## Setup checklist
- [x] `setup/claude-skills/` copied to `.claude/skills/` (2026-11-03)
- [x] Colab MCP server set up in `.mcp.json` (uv 0.13.7, server starts) (2026-11-03)
- [x] New session started; skills and `colab-mcp` visible; Colab connection tested (2026-11-03:
  connection OK; GPU runtime gave A100-SXM4-40GB, Python 3.13.16, torch 2.11.0+cu130)
- [x] Laptop venv `.venv` with Python 3.11.9 and torch 2.11.0+cu128 (RTX 3050 Ti, CUDA works) (2026-11-03)

## Results so far
All TBD. Table: `docs/04_EXPERIMENTS.md`.

## GPU hour log
| Date | Platform | GPU | Run | Hours |
|---|---|---|---|---|

## Open questions
- Weekly Kaggle GPU quota of this account (see Kaggle settings).
- Rules on external data and pretrained weights are not stated by the organizers (our plan
  does not need them).

## Log (newest first: date · task · what changed · how to run · result)
- 2026-11-03 · P0.3 · `src/raynet/check_data.py` (checks 1-7 of docs/01_DATA.md, chunked memmap
  reads, skips missing files) + `tests/test_check_data.py` (fake dataset). Data folder also has
  `manifest.json` / `subset_index.csv` (the Kaggle leaderboard is a 100-map val subset, marked
  "gameable": a format check only) and `val_terrains.txt`. No test inputs yet (December). ·
  `python -m raynet.check_data --root data/LunarRM` (about 1 min) · PASS; see
  `results/p0_check_data.txt` (for example 5.8 GHz train valid mean 89.62%, all voids = 145.0 dB).
- 2026-11-03 · P0.2 · Added remote `origin` and pushed `main`. Commits show only the owner as
  author (no AI co-author lines). · `git push` · Files visible on GitHub.
- 2026-11-03 · P0.1 · `pyproject.toml` (package `raynet`, src layout, extra `dev`),
  `src/raynet/__init__.py`, `tests/test_smoke.py`, empty folders `configs/ scripts/ notebooks/
  splits/ results/`, MIT `LICENSE`, `git init` + first commit. `KILAVUZ.pdf` and `PROMPTS_TR.md`
  are kept out of git (personal guides). · `.venv\Scripts\activate`, then
  `pip install -e ".[dev]"` and `pytest -q` · 1 test passed, ruff clean.
- 2026-11-03 · setup · Colab MCP connection tested in a new session; GPU runtime works.
- 2026-11-03 · setup · Copied skills to `.claude/skills/`. Installed uv. Added `.mcp.json`
  with the Colab MCP server (browser session mode). Owner chose Colab as the main GPU platform
  (docs/06_COMPUTE.md still says Kaggle; update it later). · Start a new session. · Connection test TBD.
