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
1. Claude Code changes code on the laptop → `pytest -q` → CPU smoke test.
2. Commit and push to GitHub. Note the commit hash.
3. On Colab: `notebooks/colab_runner.ipynb` clones the repo at that commit, copies the data
   from Drive to `/content`, and starts the run. The run folder is on Drive.
4. The run writes `last.pt` to Drive after **every epoch**. If the runtime stops, start the
   notebook again with `--resume auto`; the run continues from the last epoch.
5. Copy the run folder (without `*.pt`) from Drive to `runs/` on the laptop and tell Claude Code
   "E4_58_s0 bitti, sonuçlar runs/ içinde" → it updates `PROGRESS.md`.

## Folders on Google Drive
```
MyDrive/raynet/
  data/LunarRM.zip     the competition zip (downloaded once from Kaggle)
  runs/<run_name>/     run folders (config, metrics, last.pt, ...)
  official_runs/       checkpoints of the official baseline code (E0, P1.6)
```

## One-time setup [Owner]
1. Kaggle: accept the competition rules (already done if you downloaded the data).
2. Kaggle → Settings → API → create a token. You get either a `kaggle.json` file (username +
   key) or one API token string.
3. Colab → left bar → key icon (**Secrets**). Add:
   - `KAGGLE_USERNAME` and `KAGGLE_KEY` (the two values inside `kaggle.json`), **or**
   - `KAGGLE_API_TOKEN` (the token string).
   Turn on "Notebook access" for each secret. Never put the token in a notebook cell or in git.
4. Open `notebooks/colab_runner.ipynb` from GitHub in Colab (File → Open notebook → GitHub).
5. Run the "data to Drive (once)" cell. It downloads the zip (about 10 GB) to Drive.

## Every session
1. Runtime → Change runtime type → GPU. Choose the cheapest GPU that is fast enough
   (measured in task P2.2). Check the units per hour in Colab.
2. Run the cells in order: GPU check → mount Drive → clone repo at the commit → copy and unzip
   data to `/content/data/LunarRM` → `check_data` (quick) → smoke test → the real run.
3. Keep the browser tab open during a run. Disconnect the runtime when you are done
   (Runtime → Disconnect and delete runtime), so it does not use units.
4. Write the GPU hours in the log in `PROGRESS.md`.

## Notebook cells (template; the real file is `notebooks/colab_runner.ipynb`)
```python
# 1. GPU check
!nvidia-smi
import sys, torch; print(sys.version, torch.__version__, torch.cuda.is_available())
```
```python
# 2. Drive and the run settings
from google.colab import drive
drive.mount("/content/drive")
DRIVE = "/content/drive/MyDrive/raynet"
COMMIT = "<commit-hash>"
```
```python
# 3. code at the tested commit
!git clone https://github.com/usarrahim/raynet-lunar.git /content/raynet-lunar
%cd /content/raynet-lunar
!git checkout {COMMIT}
!pip install -q -e .
```
```python
# 4. data: Drive zip -> local disk (memmaps need a fast local disk)
!mkdir -p /content/data/LunarRM
!cp {DRIVE}/data/LunarRM.zip /content/data/
!unzip -q -o /content/data/LunarRM.zip -d /content/data/LunarRM
DATA = "/content/data/LunarRM"
```
```python
# 5. smoke test, then the real run (run folder on Drive, resume after a disconnect)
!python -m raynet.train --config configs/e4_raynet_58.yaml --data-root {DATA} --epochs 1 --limit-batches 3 --out /content/smoke --run-name smoke
!python -m raynet.train --config configs/e4_raynet_58.yaml --data-root {DATA} --seed 0 --out {DRIVE}/runs --resume auto
```
```python
# 6. evaluate the last checkpoint
!python -m raynet.evaluate --run {DRIVE}/runs/E4_58_s0 --split val --data-root {DATA}
```

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
