# Official dataset and challenge notes (read on 3 Nov 2026)

Sources:
- Challenge site: https://lunarradiomapchallenge.github.io/ (pages: index, dataset, timeline, registration, results)
- Challenge code (MIT): https://github.com/anderspearson206/lunar-radiomap-challenge
  (`README.md`, `lunar_dataset.py`, `train.py`, `evaluate.py`, `metrics.py`, `score_submission.py`,
  `notebooks/lunar_starter.ipynb`)
- Kaggle: https://www.kaggle.com/competitions/the-first-lunar-pathloss-radio-map-prediction-challenge
- Terrain generator (MIT): https://github.com/anderspearson206/LunarTerrainGenerator
- Organizers: Anders Pearson, Paolo Torrado, Çağkan Yapar, Joshua Smith. Discord linked on the site.

## Task
Input: 256×256 heightmap (1 m/px) + one-hot TX position. Output: 256×256 pathloss (dB) at
415 MHz (Task 1) and 5.8 GHz = 5.775 GHz (Task 2). One conditional model or two band models.

## Simulation
Sionna RT. LoS, specular reflection, diffuse (Lambertian) scattering, refraction.
**Diffraction disabled.** Max depth 100 interactions, 4×10⁹ rays per TX, Monte Carlo averaging
over solver seeds. Lunar regolith material (values not given). TX: 20 dBm, isotropic, vertical
polarization, 3 m above the local terrain. RX: isotropic probe 1 m above the terrain.
Terrains: synthetic, procedural (LunarTerrainGenerator), checked against real lunar DTMs.
Each terrain has 1 or 51 transmitters.

## Gaps and masks
415 MHz gaps linearly interpolated; 5.8 GHz gaps = 145.0 dB placeholder (11.4% of pixels overall;
4.1% in the median map, 32.6% at the 90th percentile, 78.7% in the worst map). Valid-pixel
fractions: 415 MHz mean 99.27% / median 99.35% / min 74.31%; 5.8 GHz mean 89.57% / median 96.23% /
min 7.47%. Use the supplied masks, not `pl58 == 145.0` (float16 spacing at 145 dB is 0.125,
which mislabels about 0.02%). The official README warns: training on all pixels improves the
all-pixel diagnostic but makes the ranked metric worse, because the voids are geometrically
structured.

## Files (Kaggle bundle)
`train_index.csv train_hm.npy train_pl415.npy train_mask415.npy train_pl58.npy train_mask58.npy`,
same for `val_`, plus `metadata.json`, `README.md`, `lunar_dataset.py`, `sample_submission.csv`.
Row i of `pl*.npy` = row i of `index.csv`. Heightmaps deduplicated (`hm_row`), TX stored as
`tx_row`/`tx_col`, masks bit-packed (`np.unpackbits(...).reshape(256,256)`).
`train_pl415.npy` is 3.4 GB → `mmap_mode="r"`. `metadata.json`: `heightmap_range_m`,
`pathloss_range_db` (scale about 208 dB), `splits`. The reference loader also understands two
other layouts (nested arrays, per-file).

## Splits (per band)
train 6,120 terrains (392 with 51 TX, 5,728 with 1 TX) → 25,720 maps; val (frozen) 680 terrains
(33 / 647) → 2,330; test 1,200 terrains (75 / 1,125) → 4,950, targets and masks withheld.
Split by terrain.

## Rules
Report on the frozen val split. Re-splitting the training data for model selection is allowed
but must not be reported as the validation score. Train with masks. Augmentation encouraged
(the loader has dihedral flips/rotations). Features derived from the inputs are allowed.
Not stated on the page: rules on external data and pretrained models (ask on Discord if needed).

## Metric and baseline
RMSE in dB over valid pixels, per band, 50/50. For ranking, each band's RMSE is normalized
across teams. All-pixel RMSE is diagnostic only. Baseline (RadioUNet/RadioWNet, masked loss,
hidden TEST set): two band models 4.66 / 6.28 / 5.47 dB; one conditional model
4.76 / 6.42 / 5.59 dB. Official schedule: stage 1 450 epochs, stage 2 150 epochs (5.8 GHz
example); conditional model 300 epochs. RadioWNet has 13.27M parameters (10.92M + 2.35M).

## Submission
- Kaggle leaderboard = format checker only (scores a subset of val; plays no part in the ranking).
  CSV `ID,PL`, `ID = pl<band>_<sample_id>_<flat_pixel_index>`, row-major (`arr.reshape(-1)`).
- Final (December): one `.npy` per test sample per band (`pl415_<sample_id>.npy`,
  `pl58_<sample_id>.npy`), 256×256 in dB, plus trained model and inference code. Under 500 ms per
  map. Check with `python score_submission.py --submission <dir> --data-root <data> --band both --validate-only`.
- Timeline on the site (to be confirmed there): test inputs released 16 December, submission
  deadline 20 December. Top 5 teams are invited to submit a 2-page ICASSP 2027 paper.

## Licences and citation
Dataset CC BY 4.0. Challenge code MIT. RadioUNet code MIT (Ron Levie). Cite RadioLunaDiff
(Torrado, Pearson, Klein, Moscibroda, Smith, ICASSP 2026, pp. 21231-21235) when using the data;
RadioUNet (Levie et al., IEEE TWC 2021) for the baseline; Sionna (Hoydis et al.) for the ground truth.
