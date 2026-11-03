# PROGRESS

## Status
- Current phase: Phase 0 (setup)
- Next task: P0.2
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
- 2026-11-03 · P0.1 · `pyproject.toml` (package `raynet`, src layout, extra `dev`),
  `src/raynet/__init__.py`, `tests/test_smoke.py`, empty folders `configs/ scripts/ notebooks/
  splits/ results/`, MIT `LICENSE`, `git init` + first commit. `KILAVUZ.pdf` and `PROMPTS_TR.md`
  are kept out of git (personal guides). · `.venv\Scripts\activate`, then
  `pip install -e ".[dev]"` and `pytest -q` · 1 test passed, ruff clean.
- 2026-11-03 · setup · Colab MCP connection tested in a new session; GPU runtime works.
- 2026-11-03 · setup · Copied skills to `.claude/skills/`. Installed uv. Added `.mcp.json`
  with the Colab MCP server (browser session mode). Owner chose Colab as the main GPU platform
  (docs/06_COMPUTE.md still says Kaggle; update it later). · Start a new session. · Connection test TBD.
