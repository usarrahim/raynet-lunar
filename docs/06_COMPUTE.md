# 06 · Compute: laptop and Colab

Decision (2026-11-03): **Colab is the main GPU platform.** Kaggle is only the data source
(and a backup GPU if Colab units run out).

## Who does what
| Place | Use it for | Not for |
|---|---|---|
| Laptop (Windows, Claude Code) | code, unit tests, CPU/GPU smoke tests (RTX 3050 Ti, 4 GB), data checks, website. The full dataset is in `data/LunarRM/` | full training runs |
| Colab (GPU) | all real training and evaluation | editing code (edit on the laptop, push to GitHub) |
| Google Drive | one copy of the data, checkpoints, run folders | reading memmaps during training (too slow) |
| Kaggle | download the data once; backup GPU | main training |

## The loop
1. Claude Code changes code on the laptop → `pytest -q` → CPU smoke test → commit and push.
2. On Colab: `notebooks/colab_runner.ipynb` pulls the code (`main` by default; every run writes
   the exact commit hash to `git_commit.txt` and `summary.json`), gets the data to the local
   disk, and starts the run. The run folder is on Drive.
3. The run writes `last.pt` to Drive after **every epoch**. If the runtime stops, run the
   notebook again from the top; `--resume auto` continues from the last finished epoch.
4. Copy the run folder (without `*.pt`) from Drive to `runs/` on the laptop, or ask Claude Code
   to read it from Drive, and say "E3_58_s0 bitti" → it updates `PROGRESS.md`.

## Folders on Google Drive
```
MyDrive/raynet/
  data/LunarRM.zip     copy of the competition zip (made by the notebook the first time)
  runs/<run_name>/     our run folders (config, metrics, summary, last.pt, eval files)
  official_runs/       official baseline code runs (E0, P1.6)
```

## Data: nothing is uploaded from the laptop
The data cell of the notebook does this:
1. Data already on the local disk (same session) → nothing to do.
2. Zip on Drive (from an earlier session) → copy to `/content` (fast) and unzip.
3. First time → download from Kaggle (token from Colab Secrets) to `/content`, unzip, and keep a
   copy of the zip on Drive for the next sessions.
Training always reads from the local disk (`/content/data/LunarRM`), never from Drive: Drive is
too slow for memory maps.

## One-time setup [Owner]
1. Kaggle: accept the competition rules (already done if you downloaded the data).
2. Kaggle → Settings → API → create a token. You get either one API token string or a
   `kaggle.json` file (username + key).
3. Colab → left bar → key icon (**Secrets**). Add `KAGGLE_API_TOKEN` (the token string), **or**
   `KAGGLE_USERNAME` and `KAGGLE_KEY` (the two values inside `kaggle.json`).
   Turn on "Notebook access" for each secret. Never put the token in a notebook cell or in git.

## Every session
1. Open `notebooks/colab_runner.ipynb` from GitHub in Colab (File → Open notebook → GitHub →
   `usarrahim/raynet-lunar`).
2. Runtime → Change runtime type → GPU. Choose the cheapest GPU that is fast enough
   (measured in task P2.2). Check the units per hour in Colab.
3. Run the cells from the top: GPU check → Drive → code → data → data check → the section you
   need (speed test, P1.6, or our experiments).
4. Keep the browser tab open during a run. Disconnect the runtime when you are done
   (Runtime → Disconnect and delete runtime), so it does not use units.
5. Write the GPU hours in the log in `PROGRESS.md`.

## Notebook sections
| Section | What | GPU time |
|---|---|---|
| 1-4 | GPU check, Drive, code, data (+ full data check) | none (first time: download) |
| 5 | P2.2 speed test, E3 width 24 and 32, 300 steps each | about 10 min |
| 6 | P1.6 official RadioUNet, 5.8 GHz, 10 epochs, stage 1 | TBD (first epoch tells) |
| 7 | our experiments: pick `CONFIG` and `SEED`; train, then evaluate dev and val | about 1.5 h per run (target) |
| 8 | list what is on Drive | none |
Section 7 refuses to start while `configs/base.yaml` still says PROVISIONAL.

## E0 and P1.6 (official baseline code)
Facts from the official `train.py` (checked 2026-11-03, commit `fa3ef9b`):
- `--loss masked` is the default: MSE over valid pixels, divided by `mask.sum()`.
- `--init-from <path>` loads weights only (optimizer, LR schedule and epoch start fresh).
  `--init-from auto` prefers `_best.pt` (chosen on val), so **we always pass the `_last.pt`
  path by hand**.
- `--resume auto` continues from `<out>/<run-name>_last.pt`. `_last.pt` is saved every epoch.
- It always trains on ALL training terrains and validates on `val` every epoch. Our rules need:
  train without the dev terrains, and no look at `val` during training. So we start it through
  our wrapper `scripts/run_official.py`, which removes the dev terrains from the training set and
  gives the dev terrains to the script as its "val". The official code itself is not changed.
- Defaults: Adam, lr 1e-4, cosine to 1e-6, batch 16, AMP on, no gradient clip, flips + 90°
  rotations, `num_workers` 8. It needs `torchvision` (Colab has it).

```python
!git clone https://github.com/anderspearson206/lunar-radiomap-challenge.git /content/official
!cd /content/official && git checkout fa3ef9b
OUT = f"{DRIVE}/official_runs"
!python scripts/run_official.py --official /content/official -- --data-root {DATA} --band 58 --loss masked --phase firstU --epochs <0.75*N> --seed 0 --out {OUT} --run-name E0_58_s0_firstU --resume auto
!python scripts/run_official.py --official /content/official -- --data-root {DATA} --band 58 --loss masked --phase secondU --epochs <0.25*N> --seed 0 --out {OUT} --run-name E0_58_s0_secondU --init-from {OUT}/E0_58_s0_firstU_last.pt --resume auto
```
Then evaluate the stage-2 `_last.pt` with our `raynet.evaluate` (wrapper in
`src/raynet/models/radiounet_wrap.py`). Do not use the official `_best.pt`.
N comes from `configs/base.yaml` (task P2.2).

## Kaggle (backup GPU only)
If Colab units run out: a Kaggle notebook with the competition data attached
(Accelerator = GPU, Internet = On) can run the same cells. The data root is the
`/kaggle/input/*` folder that has `metadata.json`; outputs go to `/kaggle/working`.

## GPU hour log (keep it in PROGRESS.md)
| Date | Platform | GPU | Run | Hours |
|---|---|---|---|---|
| | | | | |
