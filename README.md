# SPU Neural Antenna Synthesis

AI-based synthesis of coplanar-waveguide (CPW) fed microstrip patch antennas, 1.0–7.0 GHz,
across 12 patch geometries in 4 families. Graduation research project — Technical College
of Engineering, Communication Department, Sulaymaniyah Polytechnic University.

Deployed on fly.io as the app `antenna-ai-project` in the `fra` region (see `fly.toml`).

---

## What it does

| Capability | Description |
|---|---|
| **Forward prediction** | Given patch length `Lp`, width `Wp` and one of 12 shapes, predict resonant frequency, bandwidth, S11 minimum, VSWR, gain and radiation efficiency, plus the full 1–7 GHz S11 curve. |
| **Inverse design** | Given a target frequency, a shape family and an optimisation priority, search the `(Lp, Wp)` space for the best-matching geometry. |
| **Validation** | Leave-one-out fidelity benchmark against the CST dataset, with MAE / P90 / Max / R² for every predicted quantity. |
| **AETHER assistant** | Groq-backed chat for questions about the design and the results. |

## Prediction method — and which part produces the answer

The app has two predictors and **uses both**:

1. **CST interpolation** — looks up `antenna_data_cleaned.csv` and interpolates between
   simulated points. This is the anchor.
2. **The trained Keras surrogate** (~115 MB) — blended in at `ML_BLEND_WEIGHT` (default
   **25%**) whenever the query is more than `ML_BLEND_RADIUS_MM` (default **0.60 mm**) from
   the nearest CST point.

Within 0.60 mm of a real simulation point the CST curve is used **verbatim** — it is ground
truth, so there is nothing for a surrogate to improve. Beyond it, the answer is 75% CST
interpolation + 25% neural model.

Both constants are at the top of `10_web_app.py`. Set `ML_BLEND_WEIGHT = 0.0` for pure CST
interpolation, or raise it toward `1.0` to trust the model more. The **Use the trained neural
model** checkbox toggles it per run, and the engine-mode line under each result states
exactly which combination produced that answer.

> Before this change the surrogate was loaded but effectively unused: swapping in a
> completely different network left the output bit-identical (`0.000e+00 dB` difference).
> Use **Engine Validation → "Compare CST vs. model"** to score both paths on the same
> held-out geometries and tune `ML_BLEND_WEIGHT` accordingly.

For the CST path, the estimator is chosen by what the dataset supports:

| Mode | When it applies |
|---|---|
| `CSV-exact` | The exact `(Lp, Wp)` point exists in the dataset. The simulated curve passes through **unmodified** — this is ground truth and is never altered. |
| `CSV-bilinear` | The four surrounding grid nodes all exist. Standard bilinear interpolation over the cell. |
| `CSV-plane` | A surrounding grid node is missing. First-order weighted local-plane fit. |
| `CSV-knn` | Last resort. Inverse-distance blend of the nearest points. |

The **Active engine components** panel on the home page reports which of these are
actually loaded at runtime, so the UI cannot claim a method it is not running.

### Resonance-aligned curve blending

Naively averaging S11 curves whose dips sit at *different* frequencies fills in the
resonance — the composite minimum ends up far too shallow and in the wrong place.
Measured under leave-one-out, plain weighted blending gave an **S11min MAE of 13.2 dB**.

`align_curves()` therefore shifts each contributing curve so its own dip coincides with
the interpolated target resonance *before* averaging. Measured effect:

| Metric (leave-one-out, 150 held-out geometries) | Plain blend | Resonance-aligned |
|---|---|---|
| fr MAE | 370.1 MHz | **225.2 MHz** |
| fr R² | 0.127 | **0.528** |
| S11min MAE | 12.78 dB | **1.56 dB** |
| S11min R² | −1.131 | **0.771** |
| Bandwidth MAE | 94.1 MHz | **17.1 MHz** |
| fr within ±2 % | 56.7 % | **75.3 %** |

> These figures were measured on a synthetic dataset with the same schema, because the
> CST CSV is not bundled with the repo. **Re-run the Engine Validation panel against the
> real `antenna_data_cleaned.csv` to get the authoritative numbers for your report.**

Set `RESONANCE_ALIGN = False` at the top of `10_web_app.py` to reproduce the pre-change
plain-average behaviour exactly, for comparison against previously published results.

## Repository layout

```
10_web_app.py       Streamlit app + prediction engine (UI, engine, validation)
engine_v2.py        Two-Brain 2.0 inference path (main brain + dip specialist)
antenna_viz.py      matplotlib geometry rendering for the 12 patch shapes
antenna_charts.py   interactive Plotly charts (S11/VSWR, comparison, design space, parity)
assets/             spu_logo.png (source) and spu_logo_nav.png (128 px, used by the UI)
Dockerfile          image build; pre-fetches the model at build time
fly.toml            fly.io deployment config
```

### External assets

Models and datasets are **not** in git. They live in the Hugging Face model repo
[`kara-nawzad/antenna-models`](https://huggingface.co/kara-nawzad/antenna-models)
(override with the `HF_REPO` env var) and are fetched at Docker **build** time:

| File | Size | Purpose |
|---|---|---|
| `forward_model_v2_curve.keras` | ~115 MB | Main surrogate, 2 heads → 4 scalars + 151-point curve |
| `antenna_data_cleaned.csv` | ~7.8 MB | CST simulation dataset (151 frequency columns) |
| `s11_dip_specialist.pkl` | — | Dip specialist. **Currently deleted from the Hub** — see Known issues. |

`scaler_geo.pkl` and `scaler_perf.pkl` (~1.6 KB each) *are* in git, as ordinary files.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run 10_web_app.py
```

For the AETHER assistant, copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml` and set `GROQ_API_KEY`.

## Deployment

```bash
fly deploy --build-arg HF_REPO=kara-nawzad/antenna-models
```

The Dockerfile downloads the model and dataset **during the image build**, so a cold
start does no Hub I/O at all. If the Hub is unreachable at build time the build still
succeeds and the app falls back to downloading on first request.

---

## Known issues

### 1. The scalers are committed as Git LFS pointers

`.gitattributes` routed `*.pkl` to LFS, so `scaler_geo.pkl` and `scaler_perf.pkl` are
129-byte pointer files in git. A plain `git clone` therefore yields text, not pickles,
and the app fails to start. fly.io works today only because `fly deploy` uploads your
local working tree, where LFS has already materialised them. Any git-based host
(Hugging Face Space, Streamlit Community Cloud, CI) will fail.

The Dockerfile now fails the build early with an explicit message if it sees a pointer.
To fix the repo permanently, on a machine with git-lfs installed:

```bash
git lfs install && git lfs pull
git rm --cached scaler_geo.pkl scaler_perf.pkl
git add .gitattributes scaler_geo.pkl scaler_perf.pkl
git commit -m "store scalers as plain files, not LFS pointers"
```

### 2. The dip specialist is missing, so "Two-Brain 2.0" runs one brain

`s11_dip_specialist.pkl` was deleted from the model repo (commit `36f0b05`). Without it
`used_specialist` is always `False` and the app runs the CSV-anchor path alone. Re-upload
the pickle to restore the specialist path, or relabel the engine in the UI.

`engine_v2.py` previously hardcoded a different `HF_REPO` than the rest of the app; it now
reads the same environment variable.

### 3. `use_ml_refine` was a dead parameter

The "Use ML refinement" checkbox was accepted by `forward_physics_first_v2()` but never
read, so it could not change any output. It is now wired into the >0.6 mm branch, where it
blends the surrogate into the CST anchor 75/25.

---

## Performance notes

| Fix | Measured effect |
|---|---|
| Model loaded once instead of twice | **145 MB less RSS** (measured with a 137 MB stand-in model) in a container that was capped at 1 GB |
| Model fetched at Docker build time | Removes a ~115 MB US→Frankfurt download from the cold-start critical path |
| Nav logo resized 1290×1044 → 128 px | **391,936 → 13,792 base64 chars per rerun** (28×), and it is displayed at 46×46 CSS px |
| Startup healthcheck cached | Was a `model.predict()` (~87 ms) plus two column scans on **every** rerun |
| `curve_columns()` memoised | 0.056 → 0.008 ms per call (7×), paid once per `forward_anchor` |
| Two `time.sleep(0.7)` calls removed | 1.4 s of artificial latency deleted |
| `requirements.txt` pinned | Rebuilds are reproducible; pandas 3.0 can no longer break a redeploy |
| Container memory 1 GB → 2 GB | Headroom for TensorFlow + a 115 MB model without OOM restarts |
