# RayNet: lunar pathloss radio maps (EHB420E course project)

## What this project is
- Final project for EHB420E Artificial Neural Networks (Istanbul Technical University).
  Owner: Abdurrahim Yılmaz. The course project is the priority; entering the ICASSP 2027
  lunar radio map challenge is optional.
- Task: from a 256×256 lunar heightmap (1 m/px) and a transmitter (TX) position, predict the
  256×256 pathloss (PL) map in dB at 415 MHz and 5.8 GHz (5.775 GHz).
- Our method (RayNet): physics prior (free-space loss + knife-edge term) → transmitter-centred
  polar view → ray-wise horizon scan → learned residual on top of the prior.
- Final product: trained model, measured analysis, report, and a public demo website where a
  user clicks a point on lunar terrain and the model draws the coverage map.
- Read first: `docs/00_PROJECT_BRIEF.md` (plan + hypotheses), `docs/08_ROADMAP.md` (tasks),
  `PROGRESS.md` (what is done).

## First session only
- If `.claude/skills/` does not exist yet, copy every folder in `setup/claude-skills/` to
  `.claude/skills/` (same names: next-task, explain, check, colab-run, add-result). Then tell
  the owner to start a new session, so the `/` commands appear. Keep `setup/` as a backup.

## How to work with the owner
- He writes in Turkish. Reply in simple Turkish. Code, comments, docstrings, commit messages
  and docs are in simple English (A2/B1 level, short sentences).
- He is a 4th-year EE student: good with Python, C/C++, Colab; new to PyTorch training at
  this scale. He must be able to explain every part in class. Prefer clear code over clever code.
- Work on ONE roadmap task at a time (`docs/08_ROADMAP.md`). For each task:
  1. Say the plan in Turkish (3-6 bullets) before editing files.
  2. Implement in small steps. Run tests.
  3. Update `PROGRESS.md` and tick the task in `docs/08_ROADMAP.md`.
  4. Finish with "Ne yaptık?", "Nasıl çalıştırılır?" and "Ne öğrendik?" (3-5 bullets each, Turkish).
- Ask before: changing hypotheses or the experiment plan, adding a dependency, deleting files,
  or starting anything that uses GPU hours.
- Never invent numbers. Every number written in docs or the report must come from a file in
  `runs/` or `results/`. If a result is missing, write "TBD".

## Science rules (do not break)
- Masked loss and masked metrics: only valid (ray-traced) pixels, mask == 1.
  Divide by `mask.sum()`, not by the number of pixels.
- The frozen validation split `val` is ONLY for final reporting.
  Tuning and decisions use our dev split: 5% of TRAINING terrains, terrain-level, fixed seed,
  saved in `splits/dev_terrains.txt`.
- No early stopping: every model is trained for the fixed budget and we evaluate its LAST
  checkpoint. (The official baseline code picks "best" on val; for E0 we use its `_last.pt`.)
- Splits are by terrain. 51 transmitters can share one terrain; never split them apart.
- Fair comparison: trained experiments share the budget in `configs/base.yaml`
  (epochs, batch size, optimizer, LR schedule, augmentation) unless the experiment tests that.
- Organizer baseline numbers (4.66 / 6.28 / 5.47 dB) are hidden TEST-set numbers after
  450+150 epochs. Do not compare our val numbers with them as if equal. Our reference is our E0.
- The dataset ray tracer has diffraction DISABLED. The knife-edge term is a shadow-depth
  feature, not ground-truth physics. Say this clearly in the report.

## Code rules
- Python >= 3.10, PyTorch >= 2.2. Package code in `src/raynet/`. Notebooks only call package code.
- Inference path must export to ONNX (the website runs the model in the browser):
  - Do NOT use `torch.cummax` in `forward` (export fails). Use `raynet.physics.scan_max_exclusive`
    (log-step scan, tested equal to cummax). See `docs/02_PHYSICS.md`.
  - Polar resampling with `F.grid_sample(..., align_corners=True)`.
  - No Python `if`/loops that depend on tensor VALUES inside `forward`.
  - Export with `torch.onnx.export(..., dynamo=True, opset_version=18)` and save as ONE file
    (no external `.onnx.data`), then run the ONNX parity test.
- Units: metres, Hz, dB. PL is positive dB. Received power RSS_dBm = 20 - PL_dB.
  TX antenna 3 m above terrain, RX 1 m above terrain.
- Coordinates: (row, col) = (y, x). Pixel centres are integer coordinates. TX is at `[tx_row, tx_col]`.
- Read normalization ranges from the dataset `metadata.json`; never hard-code them.
- Seeds: set `random`, `numpy`, `torch`. Log seed, config and git commit hash in every run.
- Each run writes `runs/<run_name>/`: `config.yaml`, `metrics.csv` (per epoch), `summary.json`,
  `last.pt` (gitignored). Evaluation writes `val_per_map.csv` (one row per map) and `val_summary.json`.
- Owner's laptop is Windows: use `pathlib`, `num_workers=0` locally, no bash-only syntax in Python.
- Never commit: `data/`, `runs/`, `*.pt`, `*.npy`, `kaggle.json`, API keys.
- Keep functions small, typed, with a one-line docstring. Add a pytest for every new module.

## Where things run
- Laptop (Windows): coding, unit tests, CPU smoke tests on a small data subset, website.
- Colab notebook with GPU (`notebooks/colab_runner.ipynb`): all real training. It clones this repo
  from GitHub, copies the data from Drive to `/content`, and runs `python -m raynet.train ...`
  with the run folder on Drive and `--resume auto`. Kaggle is only the data source and a backup
  GPU. Details: `docs/06_COMPUTE.md`.
- The laptop has the full dataset in `data/LunarRM/` (train + val, 9 GB).
- Before any GPU run, do a CPU smoke test (`--limit-batches 3 --epochs 1`).

## Commands (update this list when you add commands)
- Laptop venv (Python 3.11): `py -3.11 -m venv .venv`, then `.venv\Scripts\activate`
  (install torch first, then the package)
- Install (dev): `pip install -e ".[dev]"`
- Tests: `pytest -q`
- Lint: `ruff check src tests`
- Data check: `python -m raynet.check_data --root data/LunarRM`
- Train: `python -m raynet.train --config configs/e3_residual_58.yaml --data-root <root>`
- Evaluate: `python -m raynet.evaluate --run runs/<name> --split val --data-root <root>`
- Export ONNX: `python -m raynet.export_onnx --run runs/<name> --out web/public/models/raynet.onnx`
- Website: `cd web` then `npm install` and `npm run dev`

## Docs map (read the matching doc before a task)
- `docs/01_DATA.md`: files, loader, splits, sanity checks
- `docs/02_PHYSICS.md`: formulas, polar transform, horizon scan, test vectors, acceptance tests
- `docs/03_MODEL.md`: models E0-E6, shapes, parameter budget
- `docs/04_EXPERIMENTS.md`: experiment table, budget, seeds, run names, results table
- `docs/05_EVALUATION.md`: metrics, LoS/NLOS/border regions, statistics, timing
- `docs/06_COMPUTE.md`: Colab workflow (Kaggle backup), GPU budget
- `docs/07_WEBSITE.md`: demo website specification
- `docs/08_ROADMAP.md`: ordered tasks with acceptance criteria
- `docs/09_REPORT_GUIDE.md`: final report and presentation
- `docs/reference/`: official dataset notes, verified physics prototype, test vectors, HW1 proposal
- `setup/claude-skills/`: source of the project skills (copied to `.claude/skills/` in the first session)
