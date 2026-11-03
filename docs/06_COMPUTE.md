# 06 · Compute: laptop, Kaggle, Colab

## Who does what
| Place | Use it for | Not for |
|---|---|---|
| Laptop (Windows, Claude Code) | writing code, unit tests, CPU smoke tests on `data/mini` or val files, website | full training |
| Kaggle notebook (GPU) | all real training and evaluation; data is already attached there | editing code (edit on the laptop, push to GitHub) |
| Colab Pro | backup GPU when the Kaggle quota is used up | storing data long term |
If the laptop has an NVIDIA GPU (`nvidia-smi` works), short local tests on GPU are fine too.

## The loop
1. Claude Code writes or changes code on the laptop → `pytest -q` → CPU smoke test.
2. Commit and push to GitHub (public repo, MIT licence).
3. On Kaggle: the runner notebook clones the repo at that commit and starts the run.
4. Download the run folder (logs + `last.pt`) from the notebook Output, put logs in `runs/`,
   tell Claude Code "E4_58_s0 bitti, sonuçlar runs/ içinde" → it updates `PROGRESS.md`.

## Kaggle setup (once)
1. Join the competition and accept the rules (needed to attach the data).
2. Verify your phone number in Kaggle settings (needed for internet in notebooks).
3. New Notebook → "Add Input" → the competition data.
4. Settings: Accelerator = GPU (T4 ×2 or P100), Internet = On, Persistence = Files (optional).

## Kaggle runner notebook (`notebooks/kaggle_runner.ipynb`), one cell each
```python
# 1. code (pin the commit you tested)
!git clone https://github.com/<github-user>/raynet-lunar.git
%cd raynet-lunar
!git checkout <commit-hash>
!pip install -q -e .
```
```python
# 2. find the data root (the folder that has metadata.json)
import glob, os
DATA = [p for p in glob.glob("/kaggle/input/*") if os.path.exists(f"{p}/metadata.json")][0]
print(DATA)
```
```python
# 3. smoke test first (2 minutes), then the real run
!python -m raynet.train --config configs/e4_raynet_58.yaml --data-root {DATA} --epochs 1 --limit-batches 5 --out /kaggle/working/runs --run-name smoke
!python -m raynet.train --config configs/e4_raynet_58.yaml --data-root {DATA} --seed 0 --out /kaggle/working/runs --resume auto
```
```python
# 4. evaluate on val and pack the results
!python -m raynet.evaluate --run /kaggle/working/runs/E4_58_s0 --split val --data-root {DATA}
!cd /kaggle/working && zip -r E4_58_s0.zip runs/E4_58_s0 -x "*.pt"
```
- For long runs use **"Save Version → Save & Run All (Commit)"**. It runs in the background,
  so you can close the browser. Files in `/kaggle/working` become the version's Output.
- Keep each run well under the session time limit shown by Kaggle; use `--resume auto` to
  continue from `last.pt` in a new session (upload `last.pt` as a private Kaggle Dataset).
- Save useful checkpoints as a private Kaggle Dataset (for example `raynet-checkpoints`), so
  evaluation and the website export can load them later.

## E0 on Kaggle (official baseline code)
```python
!git clone https://github.com/anderspearson206/lunar-radiomap-challenge.git /kaggle/working/official
%cd /kaggle/working/official
!python train.py --data-root {DATA} --band 58 --loss masked --phase firstU  --epochs <0.75*N> --seed 0
!python train.py --data-root {DATA} --band 58 --loss masked --phase secondU --epochs <0.25*N> --seed 0 --init-from runs/<firstU-run-name>_last.pt
```
Then evaluate the stage-2 `_last.pt` with our `raynet.evaluate` (wrapper in
`src/raynet/models/radiounet_wrap.py`). Do not use the official `_best.pt`: it was chosen on val.
N comes from `configs/base.yaml` (task P2.2).

## Colab (backup)
1. Kaggle → Settings → API → "Create New Token" → `kaggle.json`. Never commit this file.
2. In Colab: upload `kaggle.json`, then
   ```
   !pip install -q kaggle && mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   !kaggle competitions download -c the-first-lunar-pathloss-radio-map-prediction-challenge -p /content/data
   !cd /content/data && unzip -q '*.zip'
   ```
   (about 10 GB; to save time download only the files you need with `-f <file>`).
3. Keep data on the Colab local disk (`/content`), not on Google Drive (Drive is slow for memmaps).
4. Save `runs/` to Google Drive at the end of the session.
5. Choose the cheapest GPU that is fast enough (usually T4). Check the units per hour in Colab.

## GPU hour log (keep it in PROGRESS.md)
| Date | Platform | GPU | Run | Hours |
|---|---|---|---|---|
| | | | | |
