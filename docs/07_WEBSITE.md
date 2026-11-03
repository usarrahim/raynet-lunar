# 07 · Demo website

## Goal
A public web page. A visitor:
1. picks a lunar terrain from a gallery,
2. clicks a point to place the transmitter (TX),
3. picks the band (415 MHz or 5.8 GHz),
4. sees the coverage map predicted by OUR model in about 1-2 seconds:
   received power (dBm) or pathloss (dB), a coverage threshold slider (default −100 dBm),
   "covered area %" for the tile, the geometric shadow outline, and a hover read-out
   (x, y, height, PL, RSS, LoS),
5. on validation terrains at the dataset TX positions: compares with the ray-traced ground
   truth and sees the error map,
6. reads short "How it works", "Results" and "About" sections.

Starting point: `web/prototype/index.html` (made on 3 Nov 2026). It is a single-file, working
preview with generated terrains and a JavaScript version of the physics prior (E1). There is
no trained model in it. Reuse its UI, colour map and legend code. Replace the JavaScript physics
engine with the ONNX model.

## Architecture (static site, no server)
- Vite + TypeScript, no UI framework (plain DOM + canvas). Build output `web/dist`.
- Hosting: GitHub Pages via GitHub Actions. Vite `base: "/raynet-lunar/"`.
- Model in the browser: **onnxruntime-web, pinned to 1.30.0** (tested), WASM execution provider.
  Run inference in a Web Worker so the page does not freeze. WebGPU is optional later.
- GitHub Pages does not send COOP/COEP headers, so WASM runs single-threaded. That is fine.
  (Optional: `coi-serviceworker` for threads.)

### ONNX contract (the export script must produce exactly this)
| Name | Type / shape | Meaning |
|---|---|---|
| input `height` | float32 [1,1,256,256] | metres above the tile minimum (raw, not normalized) |
| input `tx` | float32 [1,2] | (row, col) |
| input `band` | float32 [1] | 0 = 415 MHz, 1 = 5.8 GHz |
| output `pathloss_db` | float32 [1,1,256,256] | prediction in dB |
| output `los` | float32 [1,1,256,256] | geometric line of sight (for the shadow outline) |
- Normalization and all physics features are INSIDE the graph. The browser does no physics.
- Export: `torch.onnx.export(model, args, path, dynamo=True, opset_version=18)`, then save as one
  file: `onnx.save_model(onnx.load(path, load_external_data=True), out, save_as_external_data=False)`.
  The dynamo exporter writes a separate `.onnx.data` file by default; the browser cannot load it
  without extra options.
- Tested on 3 Nov 2026: the physics prior exported this way runs in onnxruntime-web 1.30.0
  (headless Chromium, WASM, 1 thread) in about 70 ms after warm-up; max difference to Python
  onnxruntime 0.0002 dB. Set `ort.env.wasm.wasmPaths` to an absolute URL.
- Targets: model file ≤ 25 MB (use fp16 weights if needed), ≤ 2 s per map on a normal laptop.

## Data assets (`web/public/tiles/`)
- **Generated terrains:** 24 tiles (16 highland, 8 maria) from LunarTerrainGenerator (MIT),
  made by `scripts/make_web_tiles.py`.
  - Install: `pip install git+https://github.com/anderspearson206/LunarTerrainGenerator.git`
  - `gen = LTGen(size=256); gen.generate_crater_dict(max_radius=1000); hm = gen.gen_highland()`
    (`gen_maria()` for maria). With the default `max_radius` the crater dictionary needs a lot
    of RAM (a 7 GB machine was killed); `max_radius=1000` worked and took about 1-8 s per tile.
  - Subtract the tile minimum (the dataset stores metres above the tile minimum).
  - Run this script on Kaggle or Colab (Linux). Its dependency `noise` is a C extension
    that needs a compiler; on Windows that means installing C++ build tools.
- **Validation tiles:** 8 terrains from the val split (CC BY 4.0, attribution on the page), with
  their dataset TX positions, ground truth for both bands and masks → "Compare with ray tracing".
- File format: `tiles/<id>/height.f16` (raw little-endian float16, 256×256), ground truth
  `tiles/<id>/pl58_<k>.f16`, `mask58_<k>.bin` (packbits), and `tiles/index.json`
  (id, kind, source, licence, TX list). Decode float16 with a small JS function
  (do not rely on `Float16Array`). Keep all assets ≤ 30 MB.
- Stretch: "upload your own heightmap" (.npy or 16-bit PNG, 256×256, metres).

## Layout
- Header: "RayNet: Lunar Coverage Demo" and one-line subtitle.
- Left panel: terrain gallery (hillshade thumbnails), band toggle, display mode (RSS dBm / PL dB),
  threshold slider, overlay opacity, "Random TX", "Compare with ray tracing" (val tiles only),
  "Download PNG".
- Centre: canvas (tile drawn at 2×), hillshade + colour overlay, TX star, dashed shadow outline,
  colour bar with labels.
- Stats card: covered area %, mean RSS, LoS %, inference time.
- Below: How it works (3 steps with small figures), Results (table from the report),
  About (course, author, AI-use statement, data and code licences, GitHub link).
- Colour map: viridis or cividis (colour-blind safe). Gray = invalid pixels in ground truth.
- Works on a phone (360 px wide): the canvas scales to the screen width.
- Keyboard: arrow keys move the TX by 1 px.
- Stretch: 3D view with three.js (terrain mesh + coverage texture, click with raycasting).

## Behaviour details
- Canvas click (u, v) → `row = floor(v / scale)`, `col = floor(u / scale)`.
- Show the terrain at once; load the model in the background with a progress message.
- Debounce clicks (150 ms); cache results by (tile, tx, band).
- Show "model: RayNet E4 (val RMSE x.xx dB)" from `public/models/model_card.json`, which the
  export script writes (numbers from `val_summary.json`, never typed by hand).

## Tests
- Playwright smoke test: page loads, no console errors, a click draws a map within 5 s, the
  covered-area number appears.
- Parity test: for 3 fixed (tile, TX, band) cases, browser output vs PyTorch output, max
  difference < 0.05 dB (reference outputs in `web/tests/fixtures/`).

## Deployment
- `.github/workflows/pages.yml`: on push to `main` → `npm ci && npm run build` in `web/` →
  upload `web/dist` → deploy to GitHub Pages.
- Single files on GitHub must be < 100 MB; our model and tiles are far below that.

## Fallback (only if ONNX in the browser fails)
- Gradio app on Hugging Face Spaces (free CPU): `app.py` loads the PyTorch checkpoint;
  `gr.Image(...).select()` gives the click position; return the coloured map. Link it from the page.

## Milestones
- **W0 (optional, after P2.1):** site skeleton with the E1/E1b prior exported to ONNX, deployed.
  The demo works early, before any network is trained.
- **W1 (Phase 4):** RayNet ONNX, validation-tile comparison, results section.
- **W2 (stretch):** 3D view, upload your own heightmap, English/Turkish switch.
