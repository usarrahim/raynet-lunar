# 01 · Data

## Source and licence
- Kaggle competition: https://www.kaggle.com/competitions/the-first-lunar-pathloss-radio-map-prediction-challenge
  (you must accept the competition rules once before you can download or attach the data).
- Official challenge code (MIT): https://github.com/anderspearson206/lunar-radiomap-challenge
- Dataset licence: **CC BY 4.0**. Attribution text for the website and report:
  > Data: First Lunar Pathloss Radio Map Prediction Challenge dataset (Pearson, Torrado, Yapar,
  > Smith), from RadioLunaDiff (Torrado et al., ICASSP 2026), CC BY 4.0.
- Terrain generator used to make the terrains (MIT): https://github.com/anderspearson206/LunarTerrainGenerator

## Physical setup (from the official dataset page)
| Item | Value |
|---|---|
| Map | 256 × 256 px, 1 m/px (256 m × 256 m) |
| Bands | 415 MHz and 5.775 GHz ("5.8 GHz") |
| Transmitter | 20 dBm, isotropic, vertical polarization, **3 m** above local terrain |
| Receiver | isotropic probe **1 m** above terrain |
| Ray tracer | Sionna RT: LoS, specular reflection, diffuse scattering, refraction. **Diffraction disabled.** Max depth 100, 4×10⁹ rays per TX |
| Target | pathloss in dB, positive, float16. RSS_dBm = 20 - PL_dB |
| Heightmap | float32, metres above the tile minimum |
| Invalid pixels | 415 MHz: gaps linearly interpolated. 5.8 GHz: gaps filled with 145.0 dB |
| Valid fraction | 415 MHz mean 99.27% (min 74.31%). 5.8 GHz mean 89.57%, median 96.23%, min 7.47% |

## Files (Kaggle "flat-arrays" layout)
On Kaggle the data root is the `/kaggle/input/<...>` folder that contains `metadata.json`
(the official notebook finds it with a glob; do the same, do not hard-code the slug).
```
train_index.csv  train_hm.npy  train_pl415.npy  train_mask415.npy  train_pl58.npy  train_mask58.npy
val_index.csv    val_hm.npy    val_pl415.npy    val_mask415.npy    val_pl58.npy    val_mask58.npy
test_index.csv   test_hm.npy   (test targets and masks are withheld; test inputs come in December)
metadata.json    README.md     lunar_dataset.py    sample_submission.csv
```
Rules of the format (the "contract"):
- **Row i of `<split>_pl<band>.npy` is row i of `<split>_index.csv`.**
- Index columns include `sample_id`, `terrain_id`, `hm_row`, `tx_row`, `tx_col`
  (print `list(rows[0])` to see all of them).
- Heightmaps are deduplicated: `hm = hm_arr[int(row["hm_row"])]`.
- TX maps are not stored: `tx = zeros(256,256); tx[tx_row, tx_col] = 1`.
- Masks are bit-packed: `mask = np.unpackbits(mask_arr[i]).reshape(256, 256)` (1 = valid).
- Always `np.load(..., mmap_mode="r")`. `train_pl415.npy` is 3.4 GB.
- `metadata.json` has `heightmap_range_m`, `pathloss_range_db` (about 20 to 228 dB) and `splits`.
  Use these for normalization. Do not hard-code them.

Sizes (approx.): train ≈ 9 GB (both bands), val ≈ 0.8 GB, total ≈ 10 GB.

## Splits
| Split | Terrains (51 TX) | Terrains (1 TX) | Maps per band | Labels |
|---|---|---|---|---|
| train | 392 | 5,728 | 25,720 | yes |
| val (frozen) | 33 | 647 | 2,330 | yes |
| test | 75 | 1,125 | 4,950 | hidden |

Official rules we follow:
- Report scores on the frozen `val` split.
- A re-split of the training data for model selection is allowed, but it must not be reported
  as the validation score.
- Train with the validity masks. Data augmentation is encouraged. Extra input features derived
  from the inputs are allowed.

## Our dev split (for model selection)
- Take 5% of TRAINING terrains (`terrain_id`), seed 0 → `splits/dev_terrains.txt`.
- Group by terrain, so all 51 transmitters of a terrain stay together.
- Check that the dev split has some 51-TX terrains and many 1-TX terrains (similar ratio to train).
- Train on train minus dev. Dev is used for tuning (model width, number of epochs, sanity checks)
  and for the decision gates. No early stopping: we evaluate the last checkpoint.
- After the final choice, evaluate once on `val` and report that.

## Loader plan (`src/raynet/data.py`)
- Wrap the official `lunar_dataset.py` (copy it into `src/raynet/third_party/` with its MIT
  notice) or write a small own loader with the same rules. Own loader is recommended, because we
  need: TX coordinates as numbers, terrain filtering (dev split), raw dB targets and masks.
- One item returns a dict:
  `{"height": (1,256,256) float32 metres, "tx": (2,) float32 (row, col), "pl": (1,256,256) float32 dB,
    "mask": (1,256,256) float32, "band": "58" or "415", "sample_id": str, "terrain_id": str}`
- Normalization happens in the model wrapper (one place), not in the loader.
- Augmentation (train only): the 8 dihedral transforms (flips + 90° rotations), applied to height,
  pl, mask, and the TX coordinates in the same way. Test: after a transform, `height[tx]` is the
  same value as before.
- Colab / Kaggle: `num_workers=2..4`, `pin_memory=True`, `persistent_workers=True`. Windows laptop:
  `num_workers=0`.

## Local development data (laptop)
- Do not download 10 GB to the laptop. For coding and CPU smoke tests, download only
  `val_index.csv`, `val_hm.npy`, `val_pl58.npy`, `val_mask58.npy`, `metadata.json`
  (≈ 0.5 GB) from the Kaggle "Data" tab into `data/LunarRM/`.
- These val files are used locally only to test that code runs. Never use them to choose models.
- For quick experiments a script `scripts/make_mini.py` can cut 200 samples into
  `data/mini/` with the same file layout.

## Sanity checks (task P1.1)
Write `python -m raynet.check_data` that prints and asserts:
1. File list and sizes; index length equals array length for every field.
2. Value ranges: heightmap ≥ 0; PL within `pathloss_range_db`; masks only 0/1.
3. Fraction of valid pixels per band (compare with the table above).
4. At 5.8 GHz, invalid pixels are exactly 145.0 dB in most cases (float16 can mislabel ~0.02%; trust the mask).
5. Pixels near the TX have the lowest PL in the map (free-space at 2 m is about 31 dB at
   415 MHz and 54 dB at 5.8 GHz; real values may differ a little, just check they are close).
6. Count terrains with 51 TX vs 1 TX.
7. Save a figure with 6 random samples: height (hillshade), PL 415, mask 415, PL 58, mask 58.
