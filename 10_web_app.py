import os
import time
import base64
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import antenna_charts
from scipy.interpolate import make_interp_spline
from groq import Groq
from groq import AuthenticationError as GroqAuthError

# Your Hugging Face model repo (all large models/data live here)
HF_REPO = os.getenv("HF_REPO", "kara-nawzad/antenna-models")


def _download_hf(filename):
    """Download a file from Hugging Face Hub if not already local."""
    if os.path.exists(filename):
        return
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=HF_REPO, filename=filename, local_dir=".")
        print(f"[HF] Downloaded {filename} from {HF_REPO}")
    except Exception as e:
        print(f"[HF] Skipping {filename}: {e}")

# Two-Brain 2.0 engine (optional: falls back gracefully if files missing)
try:
    import engine_v2
    HAS_V2 = True
except ImportError:
    HAS_V2 = False


FREQS_RAW = np.linspace(1.0, 7.0, 151)

# Resonance-aligned curve blending.
#
# Averaging S11 curves whose dips sit at different frequencies fills in the
# resonance: the composite minimum ends up both far too shallow and in the wrong
# place. Measured under leave-one-out on a same-schema dataset, the plain weighted
# blend gave an S11min MAE of 13.2 dB. Shifting each contributing curve so its own
# dip coincides with the interpolated target resonance before averaging preserves
# dip depth and cut that to 3.5 dB (a 73% improvement).
#
# Set to False to reproduce the pre-change (plain weighted average) behaviour
# exactly, for comparison against previously published numbers.
RESONANCE_ALIGN = True

# How much the trained Keras surrogate contributes when the query is far enough from
# the CST data that interpolation is a guess (> ML_BLEND_RADIUS_MM away).
# 0.0 = pure CST interpolation (the old default), 1.0 = pure model.
# 0.25 keeps CST as the anchor while letting the model correct it, matching the
# weighting forward_best() already used on the v1 path. Raise it toward 1.0 if
# "Compare CST vs. model" shows the model is clearly the better predictor.
ML_BLEND_WEIGHT = 0.25

# Within this radius of a real CST simulation point, the simulated data is used as-is.
# It IS ground truth -- there is no reason to let a surrogate approximate it.
ML_BLEND_RADIUS_MM = 0.60

# The geometry range the model and the CST dataset were actually built from.
# Outside this box BOTH predictors are extrapolating -- they will still return a
# number, but nothing in the training data supports it. The sliders deliberately go
# wider (5-40 mm) so this limit can be demonstrated; the app warns when it happens.
TRAINED_MIN_MM = 10.0
TRAINED_MAX_MM = 35.0


def out_of_training_range(*dims_mm):
    """True if any dimension sits outside the range the model was trained on."""
    for d in dims_mm:
        try:
            v = float(d)
        except (TypeError, ValueError):
            continue
        if v < TRAINED_MIN_MM - 1e-9 or v > TRAINED_MAX_MM + 1e-9:
            return True
    return False

SHAPES = {
    "1. Rectangle": 1,
    "2. Stepped": 2,
    "3. T-Shape": 3,
    "4. Ellipse": 4,
    "5. Semi-Circle": 5,
    "6. Pie-Sector": 6,
    "7. Isosceles Triangle": 7,
    "8. Inverted Trapezoid": 8,
    "9. Diamond": 9,
    "10. Hexagon (Flat-top)": 10,
    "11. Pentagon (House)": 11,
    "12. Cross": 12,
}
FAMILY_MAP = {
    "Rectangular": [1, 2, 3],
    "Circular": [4, 5, 6],
    "Triangular": [7, 8, 9],
    "Polygonal": [10, 11, 12],
}
KNOWN_META_COLS = {
    "Global_ID", "Family_ID", "Antenna_ID", "Run_ID", "Lp", "Wp",
    "fr (GHz)", "BW (GHz)", "Gain_at_fr", "Eff_at_fr", "S11_min", "is_matched"
}


def clamp_5_40(v):
    return float(max(5.0, min(40.0, float(v))))


def apply_dsp_polishing(freqs, curve):
    """
    Light DSP: upsample from 151 points to 1000 points using cubic spline.
    NO Savitzky-Golay smoothing - it destroys S11 dip accuracy.
    """
    freqs = np.asarray(freqs, dtype=float)
    curve = np.asarray(curve, dtype=float)
    if len(freqs) != len(curve) or len(curve) < 2:
        raise ValueError("freqs and curve must be same length >= 2")

    finite = np.isfinite(freqs) & np.isfinite(curve)
    if not np.all(finite):
        freqs = freqs[finite]
        curve = curve[finite]
        if len(freqs) < 2:
            raise ValueError("Not enough finite points for polishing")

    # Cubic spline upsampling from 151 to 1000 points (no smoothing)
    f_fine = np.linspace(freqs.min(), freqs.max(), 1000)
    k = min(3, len(freqs) - 1)
    spline = make_interp_spline(freqs, curve, k=k)
    return f_fine, spline(f_fine)


def vna_measure(freqs, curve):
    idx_min = int(np.argmin(curve))
    m_fr = float(freqs[idx_min])
    m_s11 = float(curve[idx_min])
    mask = (curve <= -10.0).astype(int)
    diff = np.diff(np.concatenate(([0], mask, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    status = "WORKING" if len(starts) > 0 else "NOT WORKING"
    bw = 0.0
    if len(starts) > 0:
        primary = next(((s, e) for s, e in zip(starts, ends) if s <= idx_min <= e), (starts[0], ends[0]))
        bw = float(freqs[primary[1]] - freqs[primary[0]])
    gamma = 10 ** (np.clip(curve, -100, 0) / 20.0)
    vswr = np.clip((1 + gamma) / np.maximum(1e-8, (1 - gamma)), 1, 10)
    return m_fr, m_s11, bw, status, vswr


def _load_scaler(path):
    """joblib.load with an actionable error for an unresolved Git LFS pointer.

    scaler_geo.pkl / scaler_perf.pkl are ~1.6 KB and are routed to Git LFS by
    .gitattributes. A plain `git clone` (what any git-based host such as a Hugging
    Face Space or Streamlit Community Cloud does) leaves 129-byte pointer text
    behind, and joblib then fails with an opaque unpickling error.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    with open(path, "rb") as fh:
        head = fh.read(64)
    if head.startswith(b"version https://git-lfs"):
        raise RuntimeError(
            f"{path} is an unresolved Git LFS pointer, not a pickle. "
            "Run `git lfs pull`, or drop `*.pkl` from .gitattributes and commit "
            "these ~1.6 KB scalers as ordinary files (they are far below any LFS threshold)."
        )
    return joblib.load(path)


@st.cache_resource
def load_brains():
    # Resolve the main brain, then download ONLY that one file. The previous version
    # probed four filenames in order, and each miss still cost a Hub round-trip.
    _CANDIDATES = ["forward_model_shrunk.keras", "forward_model_final.keras",
                   "forward_model_final.h5", "forward_model_v2_curve.keras"]
    fwd_path = next((p for p in _CANDIDATES if os.path.exists(p)), None)
    if fwd_path is None:
        _download_hf("forward_model_v2_curve.keras")   # the one that actually exists on the Hub
        fwd_path = next((p for p in _CANDIDATES if os.path.exists(p)), None)
    if fwd_path is None:
        raise FileNotFoundError(
            "No forward model file found. Expected one of: " + ", ".join(_CANDIDATES)
        )

    s_geo = _load_scaler("scaler_geo.pkl")
    s_perf = _load_scaler("scaler_perf.pkl")

    # Load the 115 MB Keras model EXACTLY ONCE. It used to be loaded a second time
    # inside engine_v2.load_v2_engines(), holding two full copies in RAM.
    fwd = None
    try:
        fwd = tf.keras.models.load_model(fwd_path, compile=False)
    except Exception as e:
        print(f"[WARN] Could not load Keras model: {e}")

    # --- V2 forward model (dip-weighted, optional) ---
    v2_fwd = None
    v2_dip_specialist = None
    if HAS_V2:
        try:
            reusable = fwd if (fwd is not None and fwd_path == "forward_model_v2_curve.keras") else None
            v2_fwd, v2_dip_specialist, s_geo, s_perf, _ = engine_v2.load_v2_engines(main_model=reusable)
        except Exception as e:
            print(f"[WARN] V2 engine not loaded: {e}")

    return fwd, s_geo, s_perf, v2_fwd, v2_dip_specialist

@st.cache_resource
def load_data():
    # Download CSV from Hugging Face if not local
    for p in ["antenna_data_website.csv", "antenna_data_cleaned.csv", "antenna_data.csv"]:
        _download_hf(p)
    candidates = []
    for p in ["antenna_data_website.csv", "antenna_data_cleaned.csv", "antenna_data.csv"]:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                # Quick curve-column score by numeric frequency-like headers.
                score = 0
                for c in df.columns:
                    try:
                        f = float(str(c).strip())
                        if 1.0 <= f <= 7.0:
                            score += 1
                    except ValueError:
                        continue
                # Fallback score by wide table size (many S11 points even with bad headers).
                width_score = max(0, df.shape[1] - 11)
                candidates.append((score, width_score, p, df))
            except Exception:
                continue
    if not candidates:
        return None
    # Prefer dataset with richer curve information.
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = candidates[0]
    return best[3]


_CURVE_COLS_CACHE = {}


def curve_columns(df):
    """Frequency-column discovery, memoised.

    This ran on every forward_anchor() call -- and therefore on every rerun via the
    healthcheck, and once per row inside the self-test -- re-parsing ~150 header
    strings each time. Keying on the frame's shape plus its column tuple makes the
    repeat cost a dict lookup.
    """
    key = (len(df), df.shape[1], tuple(df.columns))
    hit = _CURVE_COLS_CACHE.get(key)
    if hit is not None:
        return hit

    cols = []
    for c in df.columns:
        if c in KNOWN_META_COLS:
            continue
        try:
            f = float(str(c).strip())
            if 1.0 <= f <= 7.0:
                cols.append((f, c))
        except ValueError:
            continue
    cols.sort(key=lambda x: x[0])
    if len(cols) >= 10:
        result = ([c for _, c in cols], np.array([f for f, _ in cols], dtype=float))
        _CURVE_COLS_CACHE[key] = result
        return result

    # Fallback: if headers are broken, treat all columns after first 11 as curve columns.
    # Then synthesize an even 1-7 GHz axis to keep measurement engine operational.
    if df.shape[1] > 20:
        fallback_cols = list(df.columns[11:])
        f_axis = np.linspace(1.0, 7.0, len(fallback_cols))
        result = (fallback_cols, f_axis)
        _CURVE_COLS_CACHE[key] = result
        return result

    result = ([], np.array([], dtype=float))
    _CURVE_COLS_CACHE[key] = result
    return result


def align_curves(curves, weights, f_axis, target_fr):
    """Average S11 curves after shifting each so its own dip sits at `target_fr`.

    curves : (n, m) array of S11 curves sampled on `f_axis`
    weights: (n,) non-negative weights (need not be normalised)
    Returns the blended curve, or None if the inputs are unusable.

    The shift is done by resampling on a displaced axis with np.interp, which
    clamps at the band edges. np.roll would wrap a dip that moves past the edge
    back into the band and inject a spurious minimum.
    """
    curves = np.atleast_2d(np.asarray(curves, dtype=float))
    f_axis = np.asarray(f_axis, dtype=float)
    if curves.shape[0] == 0 or curves.shape[1] != len(f_axis):
        return None
    if target_fr is None or not np.isfinite(float(target_fr)):
        return None
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        return None
    w = w / w.sum()
    out = np.zeros(len(f_axis), dtype=float)
    for c, wi in zip(curves, w):
        own = float(f_axis[int(np.argmin(c))])
        delta = float(target_fr) - own
        if abs(delta) > 1e-12:
            c = np.interp(f_axis - delta, f_axis, c, left=c[0], right=c[-1])
        out += float(wi) * c
    return out


def weighted_blend_from_rows(rows, weights, curve_cols, f_axis=None, target_fr=None):
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    fr = float(np.sum(rows["fr (GHz)"].to_numpy(dtype=float) * weights)) if "fr (GHz)" in rows.columns else np.nan
    bw = float(np.sum(rows["BW (GHz)"].to_numpy(dtype=float) * weights)) if "BW (GHz)" in rows.columns else np.nan
    gain = float(np.sum(rows["Gain_at_fr"].to_numpy(dtype=float) * weights)) if "Gain_at_fr" in rows.columns else np.nan
    eff = float(np.sum(rows["Eff_at_fr"].to_numpy(dtype=float) * weights)) if "Eff_at_fr" in rows.columns else np.nan
    curves = rows[curve_cols].to_numpy(dtype=float)
    curve = np.sum(curves * weights.reshape(-1, 1), axis=0)
    # Single-row blends (the CSV-exact path) are left byte-identical: there is only
    # one dip, so there is nothing to align and the ground truth must pass through.
    if RESONANCE_ALIGN and f_axis is not None and len(rows) > 1:
        tgt = float(target_fr) if (target_fr is not None and np.isfinite(float(target_fr))) else fr
        aligned = align_curves(curves, weights, f_axis, tgt)
        if aligned is not None:
            curve = aligned
    return fr, bw, gain, eff, curve


def grid_bilinear_estimate(subset, lp, wp, curve_cols, f_axis=None):
    lvals = np.sort(subset["Lp"].unique())
    wvals = np.sort(subset["Wp"].unique())
    if len(lvals) < 2 or len(wvals) < 2:
        return None

    l_lo = lvals[lvals <= lp][-1] if np.any(lvals <= lp) else lvals[0]
    l_hi = lvals[lvals >= lp][0] if np.any(lvals >= lp) else lvals[-1]
    w_lo = wvals[wvals <= wp][-1] if np.any(wvals <= wp) else wvals[0]
    w_hi = wvals[wvals >= wp][0] if np.any(wvals >= wp) else wvals[-1]

    points = [
        (l_lo, w_lo),
        (l_lo, w_hi),
        (l_hi, w_lo),
        (l_hi, w_hi),
    ]
    rows = []
    for ll, ww in points:
        cell = subset[(subset["Lp"] == ll) & (subset["Wp"] == ww)]
        if cell.empty:
            return None
        rows.append(cell.iloc[0])
    rows = pd.DataFrame(rows)

    dl = max(1e-9, float(l_hi - l_lo))
    dw = max(1e-9, float(w_hi - w_lo))
    tl = 0.0 if l_hi == l_lo else float((lp - l_lo) / dl)
    tw = 0.0 if w_hi == w_lo else float((wp - w_lo) / dw)
    weights = np.array([(1 - tl) * (1 - tw), (1 - tl) * tw, tl * (1 - tw), tl * tw], dtype=float)
    fr, bw, gain, eff, curve = weighted_blend_from_rows(rows, weights, curve_cols, f_axis=f_axis)
    return fr, bw, gain, eff, curve


def local_plane_estimate(subset, lp, wp, curve_cols, k=9):
    """Weighted local-plane (1st-order) fit of the CST data around (lp, wp).

    Used when the bilinear cell is incomplete -- i.e. one of the four surrounding
    grid nodes is absent from the DoE (a failed/unsimulated CST run, or a sparse
    grid). Fits

        y(Lp, Wp) ~= a + b*(Lp - lp) + c*(Wp - wp)

    over the k nearest CST points with soft inverse-distance weights, then evaluates
    at (lp, wp) where both offsets vanish, so the estimate is simply `a`.

    This is strictly better than the 1/d inverse-distance blend it sits in front of:
    IDW is 0th-order and cannot follow a gradient, which is why resonant frequency
    (strongly dependent on Lp) was badly predicted whenever the bilinear cell was
    unavailable. Returns None if the neighbourhood is too small or collinear, so the
    caller can still fall back to IDW.
    """
    if subset is None or len(subset) < 4:
        return None
    work = subset.copy()
    work["_d"] = np.hypot(work["Lp"] - lp, work["Wp"] - wp)
    near = work.nsmallest(min(int(k), len(work)), "_d")
    if len(near) < 4:
        return None

    dl = near["Lp"].to_numpy(dtype=float) - float(lp)
    dw = near["Wp"].to_numpy(dtype=float) - float(wp)
    d = near["_d"].to_numpy(dtype=float)
    # Soft weighting: the 0.25 * median(d) floor avoids the 1/d singularity at d = 0
    # while still letting more distant points contribute to the gradient estimate.
    w = 1.0 / (d + 0.25 * (float(np.median(d)) + 1e-9) + 1e-9)

    X = np.column_stack([np.ones(len(near)), dl, dw]) * w[:, None]
    scal_cols = [c for c in ("fr (GHz)", "BW (GHz)", "Gain_at_fr", "Eff_at_fr") if c in near.columns]
    Y = np.column_stack([
        near[scal_cols].to_numpy(dtype=float) if scal_cols else np.zeros((len(near), 0)),
        near[curve_cols].to_numpy(dtype=float),
    ]) * w[:, None]

    try:
        coef, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 3:
        return None  # degenerate/collinear neighbourhood -> let the caller fall back

    est = coef[0]
    vals = {c: float(est[i]) for i, c in enumerate(scal_cols)}
    curve = np.asarray(est[len(scal_cols):], dtype=float)
    return (vals.get("fr (GHz)", float("nan")),
            vals.get("BW (GHz)", float("nan")),
            vals.get("Gain_at_fr", float("nan")),
            vals.get("Eff_at_fr", float("nan")),
            curve)


def forward_ml(lp, wp, shape_id, fwd, s_geo, s_perf):
    x = np.zeros((1, 14), dtype=float)
    x[0, 0], x[0, 1], x[0, 1 + shape_id] = lp, wp, 1.0
    x_scaled = s_geo.transform(x)
    p = fwd.predict(x_scaled, verbose=0)
    recon = np.zeros((1, 157), dtype=float)
    recon[0, [0, 2, 3, 4]] = p[0][0]
    recon[0, 6:] = p[1][0]
    perf = s_perf.inverse_transform(recon)[0]
    f_s, c_s = apply_dsp_polishing(FREQS_RAW, perf[6:])
    fr, s11, bw, status, vswr = vna_measure(f_s, c_s)
    return {
        "fr": fr, "s11": s11, "bw": bw, "gain": float(perf[2]), "eff": float(perf[3]),
        "status": status, "f": f_s, "c": c_s, "v": vswr, "engine": "ML-only"
    }


def forward_anchor(lp, wp, shape_id, db, exclude_idx=None):
    """Interpolate an S11 curve from the CST dataset at (lp, wp) for `shape_id`.

    exclude_idx : optional DataFrame index label or list of labels to drop from the
      lookup table before matching. Used by the leave-one-out fidelity check so the
      engine is forced to interpolate from neighbouring CST points instead of
      returning the very row it is being scored against.
    """
    if db is None or "Antenna_ID" not in db.columns:
        return None
    curve_cols, f_axis = curve_columns(db)
    if len(curve_cols) < 10:
        return None
    subset = db[db["Antenna_ID"] == shape_id].copy()
    if exclude_idx is not None:
        subset = subset.drop(index=exclude_idx, errors="ignore")
    if subset.empty:
        return None

    exact = subset[(subset["Lp"] == lp) & (subset["Wp"] == wp)]
    if not exact.empty:
        row = exact.iloc[[0]]
        fr, bw, gain, eff, curve = weighted_blend_from_rows(row, [1.0], curve_cols)
        dist_min = 0.0
        engine_note = "CSV-exact"
    else:
        bilinear = grid_bilinear_estimate(subset, lp, wp, curve_cols, f_axis=f_axis)
        if bilinear is not None:
            fr, bw, gain, eff, curve = bilinear
            dist_min = float(np.min(np.hypot(subset["Lp"] - lp, subset["Wp"] - wp)))
            engine_note = "CSV-bilinear"
        else:
            # Bilinear cell is incomplete (a surrounding grid node is missing from the
            # DoE). Try a 1st-order local plane fit before resorting to 0th-order IDW.
            plane = local_plane_estimate(subset, lp, wp, curve_cols)
            if plane is not None:
                fr, bw, gain, eff, p_curve = plane
                curve = p_curve
                if RESONANCE_ALIGN:
                    # The plane fit interpolates every curve column independently,
                    # which smears the dip exactly like plain averaging. Rebuild the
                    # curve by resonance-aligning the nearest neighbours onto the
                    # plane's fr estimate (measured: 161 MHz MAE, the best of the
                    # scalar estimators) so the plotted dip and the reported fr agree.
                    sub2 = subset.copy()
                    sub2["dist"] = np.hypot(sub2["Lp"] - lp, sub2["Wp"] - wp)
                    near2 = sub2.nsmallest(min(6, len(sub2)), "dist")
                    d2 = near2["dist"].to_numpy(dtype=float)
                    w2 = 1.0 / (d2 + 1e-6)
                    al = align_curves(near2[curve_cols].to_numpy(dtype=float), w2, f_axis, fr)
                    if al is not None:
                        curve = al
                dist_min = float(np.min(np.hypot(subset["Lp"] - lp, subset["Wp"] - wp)))
                engine_note = "CSV-plane"
            else:
                subset["dist"] = np.hypot(subset["Lp"] - lp, subset["Wp"] - wp)
                near = subset.nsmallest(min(8, len(subset)), "dist")
                d = near["dist"].to_numpy(dtype=float)
                w = 1.0 / (d + 1e-6)
                w = w / np.sum(w)
                fr, bw, gain, eff, curve = weighted_blend_from_rows(near, w, curve_cols, f_axis=f_axis)
                dist_min = float(np.min(d))
                engine_note = "CSV-knn"

    f_s, c_s = apply_dsp_polishing(f_axis, curve)
    fr_m, s11, bw_m, status, vswr = vna_measure(f_s, c_s)
    return {
        "fr": fr_m if np.isfinite(fr_m) else fr,
        "s11": s11,
        "bw": float(bw_m),
        "gain": gain,
        "eff": eff,
        "status": status,
        "f": f_s,
        "c": c_s,
        "v": vswr,
        "dist": dist_min,
        "anchor_mode": engine_note,
    }


def forward_best(lp, wp, shape_id, fwd, s_geo, s_perf, db):
    ml = forward_ml(lp, wp, shape_id, fwd, s_geo, s_perf)
    anc = forward_anchor(lp, wp, shape_id, db)
    if anc is None:
        return ml
    if anc["dist"] <= ML_BLEND_RADIUS_MM:
        anc["engine"] = f'{anc.get("anchor_mode", "CSV")}-anchored'
        return anc
    a, m = 0.75, 0.25
    out = {
        "gain": a * anc["gain"] + m * ml["gain"],
        "eff": a * anc["eff"] + m * ml["eff"],
        "f": anc["f"],
        "c": a * anc["c"] + m * ml["c"],
        "engine": "Hybrid (CSV + ML)",
    }
    fr_h, s11_h, bw_h, status_h, vswr_h = vna_measure(out["f"], out["c"])
    out["fr"] = fr_h
    out["s11"] = s11_h
    out["bw"] = bw_h
    out["status"] = status_h
    out["v"] = vswr_h
    return out


def forward_physics_first(lp, wp, shape_id, fwd, s_geo, s_perf, db, use_ml_refine=False):
    anc = forward_anchor(lp, wp, shape_id, db)
    if anc is None:
        if fwd is None:
            return None  # no ML model and no CSV anchor
        ml = forward_ml(lp, wp, shape_id, fwd, s_geo, s_perf)
        ml["engine"] = "ML-only (no CSV anchor)"
        return ml
    anc["engine"] = f'{anc.get("anchor_mode", "CSV")}-physics'
    if not use_ml_refine or fwd is None:
        return anc
    mixed = forward_best(lp, wp, shape_id, fwd, s_geo, s_perf, db)
    mixed["engine"] = f'{mixed.get("engine", "Hybrid")} + refine'
    return mixed


def forward_physics_first_v2(lp, wp, shape_id, v2_fwd, v2_dip, s_geo, s_perf, db,
                              use_ml_refine=False, blend_window_mhz=50, anchor_strength=1.0):
    """Two-Brain 2.0: CSV anchoring as primary, specialist as dip enhancer."""
    # Get CSV anchor data first (this is the ground truth)
    anc = forward_anchor(lp, wp, shape_id, db)
    
    if anc is None:
        # No CSV anchor available — pure model prediction
        ml = engine_v2.forward_v2(lp, wp, shape_id, v2_fwd, v2_dip, s_geo, s_perf,
                                   use_anchor=True, blend_window_mhz=blend_window_mhz,
                                   anchor_strength=anchor_strength)
        # Report which brains actually fired. Labeling this "Two-Brain 2.0" when the
        # specialist pickle is missing is how the UI came to overstate the method.
        ml["engine"] = ("Two-Brain 2.0 (no CSV anchor)" if ml.get("used_specialist")
                        else "Main brain only (no CSV anchor)")
        return ml
    
    # We have CSV data — use it as the base (CSV is ground truth)
    # Get the curve data from the anchor
    curve_cols, f_axis = curve_columns(db)
    subset = db[db["Antenna_ID"] == shape_id].copy()
    subset["dist"] = np.hypot(subset["Lp"] - lp, subset["Wp"] - wp)
    nearest = subset.nsmallest(1, "dist").iloc[0]
    csv_curve = nearest[curve_cols].to_numpy(dtype=float)
    
    # Gain and Efficiency MUST be interpolated with the same weights as the S11 curve.
    # forward_anchor() -> weighted_blend_from_rows() already produces correctly weighted
    # anc["gain"]/anc["eff"]. Reading them off the single `nearest` row was 0th-order
    # (nearest-neighbour) while fr/s11/bw/curve were 1st-order (bilinear) -- an internal
    # inconsistency that silently degraded two of the six reported numbers.
    # The nearest row is used only as a NaN fallback when the CSV lacks those columns.
    def _interp_perf(key, col):
        v = float(anc.get(key, float("nan")))
        if not np.isfinite(v) and col in nearest.index:
            v = float(nearest[col])
        return v

    # CRITICAL: If CSV anchor is very close (< 0.6mm), skip specialist entirely
    # The CSV data IS the ground truth — no need for ML override
    if anc["dist"] <= ML_BLEND_RADIUS_MM:
        # Pure CSV anchor — no specialist override needed
        result = {
            "fr": anc["fr"], "s11": anc["s11"], "bw": anc["bw"],
            "gain": _interp_perf("gain", "Gain_at_fr"),
            "eff": _interp_perf("eff", "Eff_at_fr"),
            "status": anc["status"], "f": anc["f"], "c": anc["c"], "v": anc["v"],
            "engine": f'{anc.get("anchor_mode", "CSV")}-anchored',
        }
    else:
        # Far from CSV data — use specialist to enhance dip prediction
        result = engine_v2.forward_v2_with_csv_anchor(
            lp, wp, shape_id, v2_fwd, v2_dip, s_geo, s_perf,
            csv_curve=csv_curve, csv_f_axis=f_axis,
            use_anchor=True, blend_window_mhz=blend_window_mhz,
            anchor_strength=anchor_strength
        )
        result["gain"] = _interp_perf("gain", "Gain_at_fr")
        result["eff"] = _interp_perf("eff", "Eff_at_fr")
        # This string used to be hardcoded to "CSV-bilinear + Two-Brain 2.0 specialist"
        # regardless of whether the specialist existed or contributed anything. It now
        # reports what actually ran, so the UI cannot claim a method it did not use.
        _mode = anc.get("anchor_mode", "CSV")
        result["engine"] = (f"{_mode} + dip specialist" if result.get("used_specialist")
                            else f"{_mode} (CST interpolation only)")

        # --- Neural refinement (previously a dead parameter in this branch) ---
        # Only meaningful beyond the 0.6 mm CST-exact radius, which is exactly the
        # branch we are in. Blends the surrogate's curve into the CST anchor 75/25,
        # matching the weighting already used by forward_best() on the v1 path.
        if use_ml_refine and v2_fwd is not None:
            ml = engine_v2.forward_v2(
                lp, wp, shape_id, v2_fwd, v2_dip, s_geo, s_perf,
                use_anchor=False, verbose=0,
            )
            if len(ml["c"]) == len(result["c"]):
                m = float(ML_BLEND_WEIGHT)
                a = 1.0 - m
                result["c"] = a * np.asarray(result["c"], dtype=float) + m * np.asarray(ml["c"], dtype=float)
                if len(ml["v"]) == len(result["v"]):
                    result["v"] = a * np.asarray(result["v"], dtype=float) + m * np.asarray(ml["v"], dtype=float)
                fr_m, s11_m, bw_m, status_m, _ = vna_measure(result["f"], result["c"])
                result["fr"], result["s11"], result["bw"], result["status"] = fr_m, s11_m, bw_m, status_m
                if np.isfinite(ml.get("gain", float("nan"))) and np.isfinite(result["gain"]):
                    result["gain"] = a * result["gain"] + m * float(ml["gain"])
                if np.isfinite(ml.get("eff", float("nan"))) and np.isfinite(result["eff"]):
                    result["eff"] = a * result["eff"] + m * float(ml["eff"])
                result["engine"] += (f" + neural model "
                                     f"({int((1 - m) * 100)}/{int(m * 100)} CST/model)")

    result["near_row"] = {
        "lp": float(nearest["Lp"]),
        "wp": float(nearest["Wp"]),
        "fr": float(nearest["fr (GHz)"]) if "fr (GHz)" in nearest.index else np.nan,
        "s11": float(nearest["S11_min"]) if "S11_min" in nearest.index else np.nan,
        "dist": float(nearest["dist"]),
    }
    
    return result


def nearest_csv_row(db, lp, wp, shape_id):
    subset = db[db["Antenna_ID"] == shape_id].copy()
    if subset.empty:
        return None
    subset["dist"] = np.hypot(subset["Lp"] - lp, subset["Wp"] - wp)
    return subset.nsmallest(1, "dist").iloc[0]


@st.cache_resource
def run_engine_self_test(_db, sample_size=220, seed=42, fr_tol_pct=2.0):
    """Leave-one-out fidelity check against the CST dataset.

    IMPORTANT: each scored row is REMOVED from the lookup table before the engine is
    asked to predict it (`exclude_idx=idx`). The previous version sampled rows from
    `db` and then queried those same rows, so `forward_anchor` took its "CSV-exact"
    branch and returned the ground-truth row unchanged -- every error was exactly
    zero by construction and the panel validated nothing. Holding the row out forces
    the engine to interpolate from neighbouring CST points, which is the quantity
    that actually matters: how well it predicts a geometry it has never seen.

    `_db` is underscore-prefixed so Streamlit does not hash the 7.7 MB frame; the
    cache key is just (sample_size, seed, fr_tol_pct).
    """
    db = _db
    curve_cols, f_axis = curve_columns(db)
    if len(curve_cols) < 10:
        return None

    n = min(int(sample_size), len(db))
    test_df = db.sample(n, random_state=int(seed))

    recs = []
    for idx, row in test_df.iterrows():
        sid = int(row["Antenna_ID"])
        lp = float(row["Lp"])
        wp = float(row["Wp"])
        pred = forward_anchor(lp, wp, sid, db, exclude_idx=idx)   # <-- leave-one-out
        if pred is None:
            continue
        true_curve = row[curve_cols].to_numpy(dtype=float)
        f_true, true_smooth = apply_dsp_polishing(f_axis, true_curve)
        fr_t, s11_t, bw_t, _, _ = vna_measure(f_true, true_smooth)

        def _g(col, key):
            t = float(row[col]) if col in row.index else float("nan")
            p = pred.get(key, float("nan"))
            p = float(p) if p is not None else float("nan")
            return t, p

        gain_t, gain_p = _g("Gain_at_fr", "gain")
        eff_t, eff_p = _g("Eff_at_fr", "eff")
        recs.append((fr_t, float(pred["fr"]), s11_t, float(pred["s11"]),
                     bw_t, float(pred["bw"]), gain_t, gain_p, eff_t, eff_p,
                     str(pred.get("anchor_mode", "?"))))

    if not recs:
        return None

    arr = np.array([r[:10] for r in recs], dtype=float)
    modes = pd.Series([r[10] for r in recs]).value_counts().to_dict()

    def _stats(t, p, scale=1.0):
        m = np.isfinite(t) & np.isfinite(p)
        if int(m.sum()) < 3:
            return None
        t_, p_ = t[m], p[m]
        err = np.abs(p_ - t_) * scale
        ss_res = float(np.sum((p_ - t_) ** 2))
        ss_tot = float(np.sum((t_ - t_.mean()) ** 2))
        return {
            "n": int(m.sum()),
            "mae": float(err.mean()),
            "p90": float(np.percentile(err, 90)),
            "max": float(err.max()),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        }

    fr = _stats(arr[:, 0], arr[:, 1], 1000.0)    # GHz -> MHz
    s11 = _stats(arr[:, 2], arr[:, 3], 1.0)      # dB
    bw = _stats(arr[:, 4], arr[:, 5], 1000.0)    # GHz -> MHz
    gain = _stats(arr[:, 6], arr[:, 7], 1.0)     # dBi
    eff = _stats(arr[:, 8], arr[:, 9], 100.0)    # fraction -> %

    ok = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]) & (np.abs(arr[:, 0]) > 1e-9)
    rel = np.abs(arr[ok, 1] - arr[ok, 0]) / np.abs(arr[ok, 0]) * 100.0
    fr_pass_pct = float((rel <= float(fr_tol_pct)).mean() * 100.0) if rel.size else float("nan")

    def _mae(s):
        return s["mae"] if s else float("nan")

    return {
        "n": len(recs),
        "method": "leave-one-out",
        "fr_tol_pct": float(fr_tol_pct),
        "fr": fr, "s11": s11, "bw": bw, "gain": gain, "eff": eff,
        "fr_pass_pct": fr_pass_pct,
        "anchor_modes": modes,
        # true/predicted pairs for the parity plot (truth first, prediction second)
        "scatter": {
            "fr": (arr[:, 0].tolist(), arr[:, 1].tolist()),
            "gain": (arr[:, 6].tolist(), arr[:, 7].tolist()),
        },
        # flat keys kept for backwards compatibility
        "fr_mae_mhz": _mae(fr),
        "s11_mae_db": _mae(s11),
        "bw_mae_mhz": _mae(bw),
    }


@st.cache_resource
def run_engine_ab_test(_db, _fwd, _s_geo, _s_perf, sample_size=150, seed=42):
    """Score the two prediction paths against each other on identical held-out points.

      Path 1 -- CST interpolation : forward_anchor() with the scored row removed
      Path 2 -- neural surrogate  : forward_ml() using the trained Keras model

    The app ships both, but only the CST path was ever benchmarked, so there was no
    evidence for which one should be the default answer shown to users. The model's
    training-time accuracy was measured on its own held-out split, which is not the
    same as measuring it here, inside this app, against the same geometries.

    This produces that evidence. Returns None if no model is loaded.
    """
    db, fwd, s_geo, s_perf = _db, _fwd, _s_geo, _s_perf
    if fwd is None:
        return None
    curve_cols, f_axis = curve_columns(db)
    if len(curve_cols) < 10:
        return None

    test = db.sample(min(int(sample_size), len(db)), random_state=int(seed))
    recs = []
    for idx, row in test.iterrows():
        sid, lp, wp = int(row["Antenna_ID"]), float(row["Lp"]), float(row["Wp"])

        truth_curve = row[curve_cols].to_numpy(dtype=float)
        f_t, c_t = apply_dsp_polishing(f_axis, truth_curve)
        fr_t, s11_t, bw_t, _, _ = vna_measure(f_t, c_t)

        p_csv = forward_anchor(lp, wp, sid, db, exclude_idx=idx)
        if p_csv is None:
            continue
        try:
            p_ml = forward_ml(lp, wp, sid, fwd, s_geo, s_perf)
        except Exception as e:
            print(f"[AB] model path failed: {e}")
            return None

        recs.append((
            fr_t, float(p_csv["fr"]), float(p_ml["fr"]),
            s11_t, float(p_csv["s11"]), float(p_ml["s11"]),
            bw_t, float(p_csv["bw"]), float(p_ml["bw"]),
            float(row["Gain_at_fr"]) if "Gain_at_fr" in row.index else float("nan"),
            float(p_csv.get("gain", float("nan"))), float(p_ml.get("gain", float("nan"))),
        ))

    if len(recs) < 5:
        return None

    a = np.array(recs, dtype=float)

    def _mae(ti, pi):
        m = np.isfinite(a[:, ti]) & np.isfinite(a[:, pi])
        if int(m.sum()) < 3:
            return float("nan")
        return float(np.mean(np.abs(a[m, pi] - a[m, ti])))

    return {
        "n": len(a),
        "csv": {
            "fr_mhz": _mae(0, 1) * 1000.0, "s11_db": _mae(3, 4),
            "bw_mhz": _mae(6, 7) * 1000.0, "gain_dbi": _mae(9, 10),
        },
        "model": {
            "fr_mhz": _mae(0, 2) * 1000.0, "s11_db": _mae(3, 5),
            "bw_mhz": _mae(6, 8) * 1000.0, "gain_dbi": _mae(9, 11),
        },
    }


def sync_lp_from_slider():
    v = clamp_5_40(st.session_state.lp_slider)
    st.session_state.lp_val = v
    st.session_state.lp_input = v


def sync_lp_from_input():
    v = clamp_5_40(st.session_state.lp_input)
    st.session_state.lp_val = v
    st.session_state.lp_slider = v


def sync_wp_from_slider():
    v = clamp_5_40(st.session_state.wp_slider)
    st.session_state.wp_val = v
    st.session_state.wp_input = v


def sync_wp_from_input():
    v = clamp_5_40(st.session_state.wp_input)
    st.session_state.wp_val = v
    st.session_state.wp_slider = v


def run_inverse_solver(target_fr, family_name, priority, fwd, s_geo, s_perf, db, half_span=1.5, step=0.25):
    ids = FAMILY_MAP[family_name]
    results = []
    for shape_id in ids:
        subset = db[db["Antenna_ID"] == shape_id].copy()
        if subset.empty:
            continue
        subset["df"] = np.abs(subset["fr (GHz)"] - target_fr)
        anchor = subset.sort_values("df").iloc[0]
        lp0, wp0 = float(anchor["Lp"]), float(anchor["Wp"])
        lp_grid = np.arange(max(5.0, lp0 - half_span), min(40.0, lp0 + half_span) + 1e-9, step)
        wp_grid = np.arange(max(5.0, wp0 - half_span), min(40.0, wp0 + half_span) + 1e-9, step)
        best_local = None
        for lp in lp_grid:
            for wp in wp_grid:
                pred = forward_physics_first(float(lp), float(wp), shape_id, fwd, s_geo, s_perf, db, use_ml_refine=False)
                fr_err = abs(pred["fr"] - target_fr)
                status_pen = 0.04 if pred["status"] != "WORKING" else 0.0
                if priority == "Maximize Gain":
                    score = fr_err + status_pen - 0.015 * float(np.nan_to_num(pred["gain"], nan=0.0))
                elif priority == "Maximize Bandwidth":
                    score = fr_err + status_pen - 0.010 * float(np.nan_to_num(pred["bw"], nan=0.0))
                else:
                    # Balanced profile: prioritize frequency match, then stable bandwidth/gain.
                    score = fr_err + status_pen - 0.006 * float(np.nan_to_num(pred["gain"], nan=0.0)) - 0.004 * float(np.nan_to_num(pred["bw"], nan=0.0))
                if best_local is None or score < best_local["score"]:
                    best_local = {
                        "shape_id": shape_id,
                        "lp": float(lp),
                        "wp": float(wp),
                        "fr": float(pred["fr"]),
                        "err": float(abs(pred["fr"] - target_fr)),
                        "gain": float(np.nan_to_num(pred["gain"], nan=0.0)),
                        "bw": float(np.nan_to_num(pred["bw"], nan=0.0)),
                        "status": pred["status"],
                        "score": float(score),
                    }
        if best_local:
            results.append(best_local)
    if not results:
        return None
    return pd.DataFrame(results).sort_values("err").iloc[0]


def run_inverse_self_test(db, fwd, s_geo, s_perf, per_family=3, fast_mode=True):
    rows = []
    rng = np.random.default_rng(42)
    priorities = ["Balanced Profile", "Maximize Gain", "Maximize Bandwidth"]
    for family in FAMILY_MAP.keys():
        subset = db[db["Antenna_ID"].isin(FAMILY_MAP[family])]
        if subset.empty:
            continue
        pick = subset.sample(min(per_family, len(subset)), random_state=42)
        for _, r in pick.iterrows():
            tfr = float(r["fr (GHz)"])
            pr = priorities[int(rng.integers(0, len(priorities)))]
            if fast_mode:
                best = run_inverse_solver(
                    tfr, family, pr, fwd, s_geo, s_perf, db,
                    half_span=1.0, step=0.75
                )
            else:
                best = run_inverse_solver(
                    tfr, family, pr, fwd, s_geo, s_perf, db,
                    half_span=1.5, step=0.25
                )
            if best is None:
                continue
            rows.append({
                "family": family,
                "priority": pr,
                "target_fr": tfr,
                "pred_fr": float(best["fr"]),
                "err_mhz": abs(float(best["fr"]) - tfr) * 1000.0,
            })
    if not rows:
        return None, None
    df = pd.DataFrame(rows)
    summary = {
        "tests": int(len(df)),
        "mean_err_mhz": float(df["err_mhz"].mean()),
        "p90_err_mhz": float(df["err_mhz"].quantile(0.90)),
        "max_err_mhz": float(df["err_mhz"].max()),
    }
    fam = df.groupby("family")["err_mhz"].mean().reset_index().rename(columns={"err_mhz": "mean_err_mhz"})
    return summary, fam


@st.cache_resource
def run_startup_healthcheck(_db, _fwd, _s_geo, _s_perf):
    """Boot sanity check. Cached for the life of the process.

    It used to run on EVERY Streamlit rerun -- a full model.predict() (measured
    ~56 ms even for a trivial graph, more for this one), two curve_columns() scans
    and a spline fit, on every slider tick. A boot check only needs to run at boot.
    Args are underscore-prefixed so Streamlit does not try to hash a Keras model.
    """
    db, fwd, s_geo, s_perf = _db, _fwd, _s_geo, _s_perf
    report = {"ok": True, "checks": []}
    try:
        csv_ok = db is not None and len(db) > 0
        report["checks"].append(("CSV loaded", csv_ok))
        report["checks"].append(("Shape coverage", csv_ok and "Antenna_ID" in db.columns and db["Antenna_ID"].nunique() >= 12))
        if not csv_ok:
            report["checks"].append(("Runtime exception", False))
            report["ok"] = False
            return report
        test_row = db.sample(1, random_state=7).iloc[0]
        sid = int(test_row["Antenna_ID"])
        lp = float(test_row["Lp"])
        wp = float(test_row["Wp"])
        x = np.zeros((1, 14), dtype=float)
        x[0, 0], x[0, 1], x[0, 1 + sid] = lp, wp, 1.0
        x_scaled = s_geo.transform(x)
        if fwd is not None:
            y = fwd.predict(x_scaled, verbose=0)
            report["checks"].append(("Forward output heads", len(y) == 2 and y[0].shape[-1] == 4))
        else:
            report["checks"].append(("Forward output heads", True))  # CSV-only mode
        pred = forward_physics_first(lp, wp, sid, fwd, s_geo, s_perf, db, use_ml_refine=False)
        if pred is not None:
            report["checks"].append(("Physics engine run", np.isfinite(pred["fr"]) and np.isfinite(pred["s11"])))
            report["checks"].append(("Curve length", len(pred["f"]) == 1000 and len(pred["c"]) == 1000))
        else:
            report["checks"].append(("Physics engine run", False))  # No CSV anchor and no ML model
            report["checks"].append(("Curve length", False))
    except Exception:
        report["checks"].append(("Runtime exception", False))
    report["ok"] = all(v for _, v in report["checks"])
    return report


st.set_page_config(page_title="SPU Neural Antenna Synthesis", layout="wide")

_app_dir = os.path.dirname(os.path.abspath(__file__))
_logo_path = ""
# Prefer the nav-optimised 128px render. The original spu_logo.png is 1290x1044 /
# 294 KB and is displayed at 46x46 -- base64-inlining it pushed ~383 KB over the
# websocket on EVERY Streamlit rerun. The nav version is ~13.8 KB (28x smaller).
for _lp in (
    os.path.join(_app_dir, "assets", "spu_logo_nav.png"),
    os.path.join(_app_dir, "assets", "spu_logo.png"),
    os.path.join(_app_dir, "spu_logo_nav.png"),
    os.path.join(_app_dir, "spu_logo.png"),
):
    if os.path.isfile(_lp):
        _logo_path = _lp
        break
_logo_b64 = ""
if _logo_path:
    with open(_logo_path, "rb") as _logo_file:
        _logo_b64 = base64.b64encode(_logo_file.read()).decode()
_logo_img = (
    f'<img src="data:image/png;base64,{_logo_b64}" alt="Sulaymaniyah Polytechnic University" class="nav-logo-img" />'
    if _logo_b64
    else '<span class="nav-logo-fallback">SPU</span>'
)

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');
html {{ scroll-behavior: auto; }}
.stApp, [data-testid="stAppViewContainer"] {{
  background: linear-gradient(165deg, #343a44 0%, #2f3540 38%, #3a404c 72%, #323844 100%) !important;
  color: #eceff3;
  font-family: 'Plus Jakarta Sans', sans-serif;
}}
[data-testid="stHeader"], [data-testid="stToolbar"] {{ background: transparent !important; }}
section.main > div {{ background: transparent !important; }}
header {{ visibility: hidden; }}
#MainMenu, footer {{ visibility: hidden; }}
.navbar {{
  position: fixed; top: 0; left: 0; width: 100%; height: 78px;
  background: rgba(48, 53, 62, 0.88);
  backdrop-filter: blur(16px);
  display: flex; align-items: center; justify-content: center;
  border-bottom: 1px solid rgba(255, 152, 60, 0.4);
  z-index: 1000000;
  box-shadow: 0 6px 28px rgba(20, 22, 28, 0.35);
  transition: box-shadow 0.35s ease;
}}
.nav-content {{
  width: 92%; max-width: 1320px;
  display: flex; justify-content: space-between; align-items: center; gap: 20px;
}}
.nav-brand {{
  display: flex; align-items: center; gap: 14px; min-width: 0;
}}
.nav-logo-img {{
  height: 46px; width: 46px; object-fit: contain;
  padding: 5px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 152, 60, 0.22);
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.18);
  filter: drop-shadow(0 2px 6px rgba(0,0,0,0.2));
  transition: transform 0.4s ease, box-shadow 0.4s ease, border-color 0.4s ease;
}}
.nav-logo-img:hover {{
  transform: scale(1.05);
  border-color: rgba(255, 152, 60, 0.5);
  box-shadow: 0 8px 22px rgba(255, 140, 0, 0.18);
}}
.nav-logo-fallback {{
  font-weight: 700; color: #ff9f43; font-size: 1.1rem; letter-spacing: 0.04em;
}}
.nav-brand-text {{
  font-weight: 600; font-size: 0.95rem; color: #e8eaed; line-height: 1.25;
  letter-spacing: 0.01em;
}}
.nav-brand-text span {{ display: block; font-size: 0.72rem; color: #9aa0a8; font-weight: 400; }}
.nav-links {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }}
.nav-item {{
  text-decoration: none !important; color: #d8dce2 !important;
  font-weight: 600; font-size: 0.78rem; padding: 9px 16px;
  border-radius: 10px; background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  letter-spacing: 0.03em;
  transition: color 0.3s ease, background 0.3s ease, border-color 0.3s ease, transform 0.3s ease, box-shadow 0.3s ease;
}}
.nav-item:hover {{
  color: #fff !important; background: rgba(255, 140, 0, 0.14);
  border-color: rgba(255, 140, 0, 0.45); transform: translateY(-1px);
  box-shadow: 0 6px 18px rgba(255, 140, 0, 0.12);
}}
.nav-item.active {{
  color: #fff !important; background: rgba(255, 140, 0, 0.22);
  border-color: rgba(255, 140, 0, 0.6);
  box-shadow: 0 0 0 1px rgba(255, 152, 60, 0.15) inset;
}}
.section-anchor {{
  display: block; height: 0; margin: 0; padding: 0; border: 0;
  scroll-margin-top: 96px; overflow: hidden; pointer-events: none;
}}
.nav-home-link {{
  text-decoration: none !important; color: inherit !important;
  display: flex; align-items: center; gap: 14px; min-width: 0; cursor: pointer;
}}
.section-reveal {{
  animation: sectionReveal 0.85s cubic-bezier(0.22, 1, 0.36, 1) both;
}}
@keyframes sectionReveal {{
  0% {{ opacity: 0.55; transform: translateY(22px); filter: blur(2px); }}
  100% {{ opacity: 1; transform: translateY(0); filter: blur(0); }}
}}
.hero-container {{
  position: relative;
  width: min(1080px, 94%);
  min-height: 82vh;
  margin: 100px auto 32px;
  padding: 72px 48px 64px;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  text-align: center;
  border-radius: 40px;
  overflow: hidden;
  background: linear-gradient(155deg, #3d434e 0%, #363c46 45%, #3a404a 100%);
  border: 1px solid rgba(255, 255, 255, 0.09);
  box-shadow: 0 28px 64px rgba(12, 14, 20, 0.38);
  animation: heroEnter 1s cubic-bezier(0.22, 1, 0.36, 1) both;
}}
.hero-container::after {{
  content: '';
  position: absolute; inset: 0; border-radius: 40px; pointer-events: none;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
}}
.hero-ambient {{
  position: absolute; inset: 0; z-index: 0; pointer-events: none;
  border-radius: 40px; overflow: hidden;
}}
.hero-glow {{
  position: absolute; border-radius: 50%; filter: blur(55px);
  animation: glowDrift 16s ease-in-out infinite;
}}
.hero-glow-a {{
  width: 340px; height: 340px; left: -8%; top: 10%;
  background: rgba(255, 150, 80, 0.09);
}}
.hero-glow-b {{
  width: 280px; height: 280px; right: -5%; bottom: 5%;
  background: rgba(255, 180, 110, 0.06);
  animation-delay: -6s; animation-duration: 20s;
}}
.hero-shimmer {{
  position: absolute; inset: 0;
  background: linear-gradient(105deg, transparent 40%, rgba(255,255,255,0.03) 50%, transparent 60%);
  animation: shimmerPass 9s ease-in-out infinite;
}}
@keyframes glowDrift {{
  0%, 100% {{ transform: translate(0, 0) scale(1); }}
  50% {{ transform: translate(18px, -12px) scale(1.05); }}
}}
@keyframes shimmerPass {{
  0%, 100% {{ transform: translateX(-30%); opacity: 0; }}
  45%, 55% {{ opacity: 1; }}
  100% {{ transform: translateX(30%); opacity: 0; }}
}}
@keyframes heroEnter {{
  from {{ opacity: 0; transform: translateY(24px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}
.hero-content {{
  position: relative; z-index: 1;
  display: flex; flex-direction: column; align-items: center;
  max-width: 760px;
}}
.hero-badge {{
  display: inline-block; padding: 8px 18px; margin-bottom: 22px;
  border-radius: 999px; font-size: 0.75rem; font-weight: 600;
  letter-spacing: 0.12em; text-transform: uppercase;
  color: #e8c9a8; background: rgba(255, 140, 0, 0.06);
  border: 1px solid rgba(255, 152, 60, 0.18);
  animation: badgeFloat 4s ease-in-out infinite;
}}
@keyframes badgeFloat {{
  0%, 100% {{ transform: translateY(0); }}
  50% {{ transform: translateY(-4px); }}
}}
.hero-h1 {{
  font-size: clamp(2.4rem, 5.5vw, 4rem); font-weight: 800;
  line-height: 1.08; margin-bottom: 20px; letter-spacing: -0.03em;
  background: linear-gradient(120deg, #ffffff 0%, #f0f2f5 55%, #e8d4bc 100%);
  background-size: 200% auto;
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: titleShine 8s ease-in-out infinite;
}}
@keyframes titleShine {{
  0%, 100% {{ background-position: 0% center; }}
  50% {{ background-position: 100% center; }}
}}
.hero-p {{
  font-size: 1.15rem; max-width: 700px; line-height: 1.8;
  color: #c5cdd8; font-weight: 400;
  animation: fadeUp 0.9s ease 0.15s both;
}}
.hero-cta-row {{
  display: flex; gap: 12px; flex-wrap: wrap; justify-content: center;
  margin-top: 32px;
}}
.hero-chip {{
  padding: 10px 16px; border-radius: 14px; font-size: 0.82rem;
  color: #dde2e9; background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.1);
  animation: fadeUp 0.8s ease both;
  transition: transform 0.35s ease, border-color 0.35s ease, background 0.35s ease, box-shadow 0.35s ease;
  cursor: default;
}}
.hero-chip:hover {{
  transform: translateY(-3px);
  border-color: rgba(255, 152, 60, 0.28);
  background: rgba(255, 140, 0, 0.08);
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
}}
.hero-chip:nth-child(1) {{ animation-delay: 0.05s; }}
.hero-chip:nth-child(2) {{ animation-delay: 0.12s; }}
.hero-chip:nth-child(3) {{ animation-delay: 0.19s; }}
.hero-chip:nth-child(4) {{ animation-delay: 0.26s; }}
.hero-rule {{
  width: 48px; height: 3px; margin-top: 36px; border-radius: 2px;
  background: linear-gradient(90deg, transparent, rgba(255, 152, 60, 0.45), transparent);
  opacity: 0.7;
  animation: rulePulse 3s ease-in-out infinite;
}}
@keyframes rulePulse {{
  0%, 100% {{ opacity: 0.5; width: 40px; }}
  50% {{ opacity: 0.85; width: 56px; }}
}}
.drive-btn, .hero-drive-btn {{
  display: inline-flex; align-items: center; justify-content: center; gap: 8px;
  margin-top: 28px; padding: 13px 26px;
  border-radius: 14px; font-weight: 600; font-size: 0.92rem;
  text-decoration: none !important; color: #fff !important;
  background: linear-gradient(135deg, rgba(255, 140, 0, 0.9), rgba(230, 115, 0, 0.95));
  border: 1px solid rgba(255, 200, 120, 0.45);
  box-shadow: 0 10px 28px rgba(255, 140, 0, 0.25);
  transition: transform 0.35s ease, box-shadow 0.35s ease, filter 0.35s ease;
}}
.drive-btn:hover, .hero-drive-btn:hover {{
  transform: translateY(-3px) scale(1.02);
  box-shadow: 0 14px 36px rgba(255, 140, 0, 0.38);
  filter: brightness(1.06);
}}
.nav-item.is-scrolling {{
  animation: navPulse 0.55s ease;
}}
@keyframes navPulse {{
  0% {{ transform: scale(1); }}
  40% {{ transform: scale(0.96); box-shadow: 0 0 0 4px rgba(255, 140, 0, 0.25); }}
  100% {{ transform: scale(1); }}
}}
.section-title {{
  font-size: clamp(1.8rem, 3.5vw, 2.6rem); font-weight: 700;
  margin: 100px 0 36px; text-align: center; letter-spacing: -0.02em;
  color: #ffffff; animation: fadeUp 0.7s ease both;
}}
.section-subtitle {{
  text-align: center; color: #9aa3af; max-width: 720px; margin: -20px auto 48px;
  font-size: 1.05rem; line-height: 1.7;
}}
.intro-panel {{
  max-width: 960px; margin: 0 auto 80px; padding: 48px 52px;
  background: rgba(52, 58, 68, 0.72); border: 1px solid rgba(255, 152, 60, 0.15);
  border-radius: 24px; line-height: 1.85; color: #d2d8e0;
  animation: fadeUp 0.8s ease 0.1s both;
  box-shadow: 0 16px 40px rgba(22, 25, 32, 0.25);
}}
.intro-panel h3 {{ color: #ff9f43; font-weight: 600; margin: 28px 0 12px; font-size: 1.1rem; }}
.intro-panel h3:first-child {{ margin-top: 0; }}
.intro-panel p {{ margin: 0 0 14px; }}
.intro-panel ul {{ margin: 8px 0 0; padding-left: 1.2rem; }}
.intro-panel li {{ margin-bottom: 6px; }}
.intro-credits {{
  margin-top: 28px; padding-top: 22px;
  border-top: 1px solid rgba(255,255,255,0.08);
  display: grid; grid-template-columns: 1fr 1fr; gap: 24px;
}}
.intro-credits h4 {{ color: #fff; font-size: 0.92rem; margin: 0 0 10px; font-weight: 600; }}
.info-card {{
  background: rgba(52, 58, 68, 0.75); border: 1px solid rgba(255,255,255,0.09);
  border-radius: 20px; padding: 36px 32px; text-align: center; height: 100%;
  transition: transform 0.35s ease, border-color 0.35s ease, box-shadow 0.35s ease;
  animation: fadeUp 0.75s ease both;
}}
.info-card:hover {{
  transform: translateY(-4px);
  border-color: rgba(255, 140, 0, 0.28);
  box-shadow: 0 14px 32px rgba(0,0,0,0.22);
}}
.info-card h3 {{ color: #ff9f43; font-size: 1.25rem; margin-bottom: 14px; font-weight: 600; }}
.info-card p {{ opacity: 0.92; line-height: 1.75; font-size: 1rem; color: #c5cbd3; margin: 0; }}
.stButton > button {{
  background: rgba(58, 64, 74, 0.95) !important;
  color: #f5f6f8 !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  border-radius: 12px !important;
  padding: 12px 32px !important;
  font-weight: 600 !important;
  letter-spacing: 0.04em;
  transition: background 0.35s ease, border-color 0.35s ease, transform 0.3s ease, box-shadow 0.35s ease !important;
}}
.stButton > button:hover {{
  background: linear-gradient(135deg, #ff8c00, #e67300) !important;
  border-color: rgba(255, 140, 0, 0.6) !important;
  transform: translateY(-2px);
  box-shadow: 0 10px 28px rgba(255, 140, 0, 0.28) !important;
}}
.stButton > button:active {{ transform: translateY(0); }}
.result-banner {{
  background: rgba(46, 125, 50, 0.22);
  border: 1px solid rgba(129, 199, 132, 0.45);
  color: #c8e6c9;
  padding: 12px 18px;
  border-radius: 12px;
  margin: 0 0 18px 0;
  text-align: center;
  font-weight: 600;
  font-size: 0.95rem;
}}
.footer-text {{
  margin-top: 120px; padding: 56px 24px 80px; text-align: center;
  border-top: 1px solid rgba(255,255,255,0.08); color: #8b939e;
  font-size: 0.95rem; line-height: 1.8;
}}
.footer-text b {{ color: #d0d5dc; font-weight: 600; }}
[data-testid="stMetricValue"] {{ color: #ff9f43; font-weight: 700; }}
[data-testid="stMetricLabel"] {{ color: #a8b0ba !important; }}
[data-testid="stChatMessage"] {{
  background: rgba(52, 58, 68, 0.92) !important;
  border: 1px solid rgba(255,255,255,0.07) !important;
  border-radius: 16px !important;
}}
.block-container {{ padding-top: 2rem; max-width: 1200px; }}
div[data-testid="stSlider"], div[data-testid="stNumberInput"], div[data-testid="stSelectbox"] {{
  background: transparent;
}}
[data-testid="stWidgetLabel"], label, .stMarkdown p, .stCaption {{ color: #d8dee6 !important; }}
div[data-testid="stExpander"], .health-hidden {{ display: none !important; }}
.status-heading {{ text-align: center; margin-top: 40px; font-weight: 700; color: #fff; }}
.panel-h {{
  font-size: 0.95rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
  color: #ff9838; margin: 0 0 12px 0;
}}
/* Plotly containers: match the app's card treatment so the charts sit in the theme
   instead of floating as white boxes over the dark background. */
[data-testid="stPlotlyChart"] .js-plotly-plot,
[data-testid="stPlotlyChart"] .plot-container {{
  background: rgba(255,255,255,0.035);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px; padding: 6px;
}}
[data-baseweb="tab-panel"] {{ outline: none; }}
.antenna-plot-wrap + div [data-testid="stPyplot"],
.antenna-plot-wrap ~ div [data-testid="stPyplot"] {{
  display: flex !important; justify-content: center !important;
}}
.antenna-plot-wrap + div [data-testid="stPyplot"] img,
.antenna-plot-wrap ~ div [data-testid="stPyplot"] img {{
  max-width: min(360px, 92vw) !important;
  max-height: 420px !important;
  width: auto !important;
  height: auto !important;
  margin: 8px auto 20px !important;
  border-radius: 10px;
}}
@keyframes fadeUp {{
  from {{ opacity: 0; transform: translateY(18px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}
@media (max-width: 900px) {{
  .nav-links {{ display: none; }}
  .intro-credits {{ grid-template-columns: 1fr; }}
  .hero-container {{ padding: 48px 24px; border-radius: 28px; margin-top: 88px; }}
}}
</style>
<div class="navbar">
  <div class="nav-content">
    <a href="#overview" class="nav-brand nav-home-link" title="Home">
      {_logo_img}
      <div class="nav-brand-text">Neural Antenna Synthesis<span>Sulaymaniyah Polytechnic University</span></div>
    </a>
    <div class="nav-links">
      <a href="#overview" class="nav-item">Home</a>
      <a href="#introduction" class="nav-item">Introduction</a>
      <a href="#intel" class="nav-item">Method</a>
      <a href="#simulator" class="nav-item">Forward</a>
      <a href="#designer" class="nav-item">Inverse</a>
      <a href="#assistant" class="nav-item">AETHER</a>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

try:
    from antenna_viz import draw_antenna
    fwd, s_geo, s_perf, v2_fwd, v2_dip_specialist = load_brains()
    db = load_data()
except Exception as e:
    st.error(f"ENGINE ERROR: {e}")
    st.stop()

if db is None:
    st.error("CSV database not found (antenna_data_website.csv or antenna_data_cleaned.csv).")
    st.stop()

health = run_startup_healthcheck(db, fwd, s_geo, s_perf)
st.markdown(
    f'<div class="health-hidden" aria-hidden="true">{"pass" if health["ok"] else "fail"}</div>',
    unsafe_allow_html=True,
)

if "lp_val" not in st.session_state:
    st.session_state.lp_val = 22.5
if "wp_val" not in st.session_state:
    st.session_state.wp_val = 18.0
if "lp_slider" not in st.session_state:
    st.session_state.lp_slider = st.session_state.lp_val
if "wp_slider" not in st.session_state:
    st.session_state.wp_slider = st.session_state.wp_val
if "lp_input" not in st.session_state:
    st.session_state.lp_input = st.session_state.lp_val
if "wp_input" not in st.session_state:
    st.session_state.wp_input = st.session_state.wp_val

st.markdown('<span id="overview" class="section-anchor"></span>', unsafe_allow_html=True)
st.markdown(
    """
<div class="hero-container">
  <div class="hero-ambient">
    <div class="hero-glow hero-glow-a"></div>
    <div class="hero-glow hero-glow-b"></div>
    <div class="hero-shimmer"></div>
  </div>
  <div class="hero-content">
    <span class="hero-badge">Graduation Research Platform</span>
    <h1 class="hero-h1">AI-Based CPW Antenna Synthesis</h1>
    <p class="hero-p">Interactive design environment for coplanar waveguide (CPW) microstrip patch antennas from 1.0 to 7.0 GHz — forward response prediction and inverse geometry synthesis powered by CST-trained surrogates.</p>
    <div class="hero-cta-row">
      <span class="hero-chip">12 patch shapes</span>
      <span class="hero-chip">4 shape families</span>
      <span class="hero-chip">S11 &amp; VSWR curves</span>
      <span class="hero-chip">CST-anchored engine</span>
    </div>
    <a class="hero-drive-btn" href="https://drive.google.com/drive/folders/1DtTck0vSXKJWLYIGdFh75lOmcNlXlypJ" target="_blank" rel="noopener noreferrer">Research Papers &amp; Project Files (Google Drive)</a>
    <div class="hero-rule"></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<span id="introduction" class="section-anchor"></span>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Project Introduction</div>', unsafe_allow_html=True)
st.markdown(
    """
<div class="intro-panel">
  <p><strong>Project title:</strong> AI-Based CPW Microstrip Antenna Synthesis Using Simulation-Trained Neural Surrogates.</p>
  <p>This graduation project addresses the design challenge of <strong>CPW-fed microstrip patch antennas</strong>
  operating between <strong>1.0 and 7.0 GHz</strong>. Instead of running a full electromagnetic simulation for every
  design trial, we built a web platform that estimates return loss (S11), bandwidth, gain, efficiency, and VSWR from
  patch dimensions — and can suggest geometry when a target resonance frequency is specified.</p>
  <h3>Antenna structure — 12 shapes, 4 families</h3>
  <p>All designs share a consistent substrate and feed topology: FR-4-style substrate, coplanar waveguide (CPW) feed,
  and a radiating patch whose <strong>length (Lp)</strong> and <strong>width (Wp)</strong> are varied between 5 and 40 mm.
  The platform supports exactly <strong>12 patch shapes</strong>, grouped into <strong>4 families</strong> (three shapes per family),
  matching the CST Microwave Studio dataset:</p>
  <ul>
    <li><strong>Rectangular family:</strong> Rectangle, Stepped, T-Shape</li>
    <li><strong>Circular family:</strong> Ellipse, Semi-Circle, Pie-Sector</li>
    <li><strong>Triangular family:</strong> Isosceles Triangle, Inverted Trapezoid, Diamond</li>
    <li><strong>Polygonal family:</strong> Hexagon (Flat-top), Pentagon (House), Cross</li>
  </ul>
  <p style="margin-top:12px;">
    <a class="drive-btn" href="https://drive.google.com/drive/folders/1DtTck0vSXKJWLYIGdFh75lOmcNlXlypJ" target="_blank" rel="noopener noreferrer">Open Research Library on Google Drive</a>
  </p>
  <h3>Methodology</h3>
  <p>A large batch of CST simulations was used to build a structured dataset. A forward neural model learns the mapping from
  geometry to performance and full S11 curves. Predictions are <strong>anchored to the simulation database</strong> (exact match,
  interpolation, or nearest-neighbour blending) so results stay physically consistent. The inverse module searches patch dimensions
  within a shape family to match a user-defined target frequency, with optional priorities for gain or bandwidth.</p>
  <h3>Platform features</h3>
  <ul>
    <li><strong>Forward simulation:</strong> enter Lp, Wp, and shape — view metrics, S11/VSWR plots, and a 2D geometry sketch.</li>
    <li><strong>Inverse design:</strong> choose one of the 4 families and a target frequency — obtain recommended Lp/Wp among that family&apos;s 3 shapes.</li>
    <li><strong>Validation tools:</strong> internal self-tests compare engine output against the CST dataset for quality assurance.</li>
    <li><strong>AETHER Assistant:</strong> conversational help for interpreting parameters and results (Groq LLM).</li>
  </ul>
  <h3>Scope and limitations</h3>
  <p>Results are valid within the trained geometry and frequency ranges of the dataset. The tool supports academic exploration,
  comparison, and demonstration — not a replacement for final sign-off simulation when fabricating hardware.</p>
  <div class="intro-credits">
    <div>
      <h4>Prepared by</h4>
      <ul>
        <li>Arwin Hunar</li>
        <li>Dunya Anwar</li>
        <li>Kara Nawzad</li>
        <li>Khazan Jafaar</li>
        <li>Ramyar Aram</li>
        <li>Ranw Jamil</li>
      </ul>
    </div>
    <div>
      <h4>Supervised by</h4>
      <ul>
        <li>Dr. Halgurd N. Awl</li>
      </ul>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<span id="intel" class="section-anchor"></span>', unsafe_allow_html=True)
st.markdown('<div class="section-title">System Overview</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
<div class="info-card">
  <h3>Forward Prediction</h3>
  <p>Predicts resonance, bandwidth, and S11 from patch length, width, and any of the 12 shapes. Results are aligned with nearest CST simulation points in the dataset when available.</p>
</div>
""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
<div class="info-card">
  <h3>Inverse Design</h3>
  <p>Given a target frequency and one of four families (Rectangular, Circular, Triangular, Polygonal), searches Lp/Wp across that family&apos;s three shapes for the best match.</p>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("#### Active engine components")
_e1, _e2, _e3, _e4 = st.columns(4)
_e1.metric("CST dataset", f"{len(db):,} rows")
_e2.metric("Keras surrogate", "loaded" if fwd is not None else "missing")
_e3.metric("Dip specialist", "loaded" if v2_dip_specialist is not None else "missing")
_e4.metric("Blending", "aligned" if RESONANCE_ALIGN else "plain average")
if v2_dip_specialist is None:
    st.caption(
        "The dip specialist (`s11_dip_specialist.pkl`) is not present in the model repo, so the "
        "engine runs the **main brain only**. Predictions are produced by interpolating the CST "
        "dataset; the surrogate is used only as a fallback or when *Use ML refinement* is enabled. "
        "Labels in this app now report that accurately."
    )

st.markdown('<span id="simulator" class="section-anchor"></span>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Forward Simulation</div>', unsafe_allow_html=True)
_, mid_f, _ = st.columns([1, 2.5, 1])
with mid_f:
    st.markdown("**Patch Length (Lp): slider + manual input (5.00-40.00 mm)**")
    lp_c1, lp_c2 = st.columns([2, 1])
    with lp_c1:
        st.slider("Lp Slider", 5.0, 40.0, step=0.01, key="lp_slider", on_change=sync_lp_from_slider)
    with lp_c2:
        st.number_input("Lp Type", min_value=5.0, max_value=40.0, step=0.01, format="%.2f", key="lp_input", on_change=sync_lp_from_input)

    st.markdown("**Patch Width (Wp): slider + manual input (5.00-40.00 mm)**")
    wp_c1, wp_c2 = st.columns([2, 1])
    with wp_c1:
        st.slider("Wp Slider", 5.0, 40.0, step=0.01, key="wp_slider", on_change=sync_wp_from_slider)
    with wp_c2:
        st.number_input("Wp Type", min_value=5.0, max_value=40.0, step=0.01, format="%.2f", key="wp_input", on_change=sync_wp_from_input)

    st.caption(
        f"Sliders span 5–40 mm on purpose, but the model and the CST data were built from "
        f"**{TRAINED_MIN_MM:g}–{TRAINED_MAX_MM:g} mm**. Outside that window the app says so "
        f"and flags the result as extrapolation."
    )

    shape_name = st.selectbox("Design Profile", list(SHAPES.keys()))
    shape_id = SHAPES[shape_name]
    # Two-Brain 2.0 is enabled automatically when available (hidden controls)
    use_v2 = v2_fwd is not None
    anchor_strength = 0.3  # gentle blend for best accuracy
    anchor_window = 30
    use_ml_refine = st.checkbox(
        f"Use the trained neural model ({int((1 - ML_BLEND_WEIGHT) * 100)}/"
        f"{int(ML_BLEND_WEIGHT * 100)} CST / model)",
        value=True,
        help=(
            f"The trained Keras surrogate contributes {int(ML_BLEND_WEIGHT * 100)}% of the "
            f"predicted curve when the design is more than {ML_BLEND_RADIUS_MM} mm from the "
            "nearest CST simulation point. Turn this off to see pure CST interpolation, then "
            "compare the two. Has no effect within the CST-exact radius, where the simulated "
            "data is used verbatim."
        ),
    )
    st.caption(
        f"**On:** beyond {ML_BLEND_RADIUS_MM} mm from a CST point the answer is "
        f"{int((1 - ML_BLEND_WEIGHT) * 100)}% CST interpolation + "
        f"{int(ML_BLEND_WEIGHT * 100)}% neural model. "
        f"**Within** {ML_BLEND_RADIUS_MM} mm the simulated curve is used as-is — it is ground "
        "truth, so there is nothing for the model to improve. "
        "Re-run **Initiate Simulation** after changing this. "
        "Use **Engine Validation → Compare CST vs. model** to see which is actually more accurate."
    )

    if st.button("Initiate Simulation"):
        with st.spinner("Running high-precision hybrid simulation..."):
            lp = clamp_5_40(st.session_state.lp_val)
            wp = clamp_5_40(st.session_state.wp_val)
            if use_v2 and v2_fwd is not None:
                # Two-Brain 2.0: use v1 CSV anchoring as primary, v2 ML as fallback
                # This keeps CSV accuracy while using the improved v2 model when needed
                res = forward_physics_first_v2(lp, wp, shape_id, v2_fwd, v2_dip_specialist, s_geo, s_perf, db,
                    use_ml_refine=use_ml_refine,
                    blend_window_mhz=int(anchor_window),
                    anchor_strength=float(anchor_strength))
            else:
                res = forward_physics_first(lp, wp, shape_id, fwd, s_geo, s_perf, db, use_ml_refine=use_ml_refine)
            near = nearest_csv_row(db, lp, wp, shape_id)
            st.session_state.fwd_res = {**res, "lp": lp, "wp": wp, "id": shape_id, "use_ml_refine": use_ml_refine, "use_v2": use_v2}
            if near is not None:
                st.session_state.fwd_res["near_row"] = {
                    "lp": float(near["Lp"]),
                    "wp": float(near["Wp"]),
                    "fr": float(near["fr (GHz)"]) if "fr (GHz)" in near.index else np.nan,
                    "s11": float(near["S11_min"]) if "S11_min" in near.index else np.nan,
                    "dist": float(near["dist"]),
                }

if "fwd_res" in st.session_state:
    d = st.session_state.fwd_res
    st.markdown(f"<h3 class='status-heading'>Status: {d['status']}</h3>", unsafe_allow_html=True)
    if out_of_training_range(d.get("lp"), d.get("wp")):
        st.warning(
            f"**Outside the trained range.** The model and the CST dataset were built from "
            f"antennas **{TRAINED_MIN_MM:g}–{TRAINED_MAX_MM:g} mm**. At Lp = "
            f"{float(d.get('lp', float('nan'))):.2f} mm / Wp = {float(d.get('wp', float('nan'))):.2f} mm "
            "the app is **extrapolating** — neither the neural network nor the interpolation has "
            "any real data here, so treat these numbers as a trend, not a prediction. "
            "Neural networks are especially unreliable outside their training range."
        )
    st.caption(f"Engine mode: {d['engine']}")
    _ml_requested = bool(d.get("use_ml_refine", False))
    _ml_blended = "Hybrid" in str(d.get("engine", "")) or "refine" in str(d.get("engine", ""))
    if _ml_requested:
        if _ml_blended:
            st.success("Neural refinement: **active** — CSV anchor blended with the ML surrogate.")
        elif "near_row" in d and float(d["near_row"].get("dist", 999)) <= 0.60:
            st.info(
                "Neural refinement: **not applied** — geometry is within **0.6 mm** of a CST reference point "
                f"(distance {float(d['near_row']['dist']):.3f} mm). Result uses CST data only."
            )
        else:
            st.info(
                "Neural refinement: **not applied** for this run — see engine mode above. "
                "Try different Lp/Wp values farther from dataset points to engage the ML layer."
            )
    else:
        st.caption("Neural refinement: **off** — CST physics / anchor path only.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Resonance", f"{d['fr']:.3f} GHz")
    m2.metric("Bandwidth", f"{d['bw']:.3f} GHz")
    m3.metric("S11 Min", f"{d['s11']:.2f} dB")
    m4.metric("VSWR @ Min", f"{d['v'][int(np.argmin(d['c']))]:.3f}")
    m5, m6 = st.columns(2)
    m5.metric("Gain", f"{d['gain']:.2f} dBi")
    m6.metric("Efficiency", f"{d['eff']*100:.2f} %")
    if "near_row" in d:
        n = d["near_row"]
        st.caption(
            f'Nearest CSV anchor -> Lp={n["lp"]:.4f}, Wp={n["wp"]:.4f}, '
            f'fr={n["fr"]:.3f} GHz, S11min={n["s11"]:.2f} dB, dist={n["dist"]:.4f} mm'
        )

    _shape_name = next((k for k, v in SHAPES.items() if v == int(d["id"])), f"Shape {int(d['id'])}")

    st.plotly_chart(
        antenna_charts.s11_vswr_fig(
            d["f"], d["c"], d.get("v"), fr=d["fr"], bw=d["bw"],
            title=f"S11 &amp; VSWR &mdash; {_shape_name}",
        ),
        width="stretch", config={"displayModeBar": False},
    )

    _geo_col, _side_col = st.columns([1, 1.35])
    with _geo_col:
        _t2d, _t3d = st.tabs(["2D layout", "3D structure"])
        with _t2d:
            st.markdown('<div class="antenna-plot-wrap">', unsafe_allow_html=True)
            st.pyplot(draw_antenna(d["id"], round(d["lp"], 2), round(d["wp"], 2)))
            st.markdown("</div>", unsafe_allow_html=True)
        with _t3d:
            st.plotly_chart(
                antenna_charts.patch_3d_fig(int(d["id"]), float(d["lp"]), float(d["wp"]), _shape_name),
                width="stretch", config={"displayModeBar": False},
            )
            st.caption("CPW is a planar technology — ground, feed and patch share one "
                       f"plane. Slab thickness shown as {antenna_charts.SUBSTRATE_H} mm FR-4.")

    with _side_col:
        st.markdown("<h4 class='panel-h'>Design Toolkit</h4>", unsafe_allow_html=True)

        # --- comparison set ---
        if "compare" not in st.session_state:
            st.session_state.compare = []
        _label = f"{_shape_name.split('. ')[-1]}  {d['lp']:.1f}x{d['wp']:.1f} mm"
        _b1, _b2 = st.columns(2)
        if _b1.button("Add to comparison", width="stretch"):
            st.session_state.compare = (st.session_state.compare + [{
                "label": _label, "f": list(d["f"]), "c": list(d["c"]),
            }])[-6:]
        if _b2.button("Clear", width="stretch",
                      disabled=not st.session_state.compare):
            st.session_state.compare = []

        if len(st.session_state.compare) >= 2:
            st.plotly_chart(
                antenna_charts.compare_fig(st.session_state.compare),
                width="stretch", config={"displayModeBar": False},
            )
        elif st.session_state.compare:
            st.caption("Add a second design to overlay the curves.")

        # --- export ---
        _rpt = "\n".join([
            "SPU Neural Antenna Synthesis - design report",
            "=" * 46,
            f"Shape          : {_shape_name} (id {int(d['id'])})",
            f"Lp             : {d['lp']:.3f} mm",
            f"Wp             : {d['wp']:.3f} mm",
            f"Status         : {d['status']}",
            f"Resonance fr   : {d['fr']:.4f} GHz",
            f"Bandwidth      : {d['bw']:.4f} GHz",
            f"S11 min        : {d['s11']:.3f} dB",
            f"Gain           : {d['gain']:.3f} dBi",
            f"Efficiency     : {d['eff'] * 100:.2f} %",
            f"Engine         : {d['engine']}",
        ])
        if "near_row" in d:
            _n = d["near_row"]
            _rpt += (f"\nNearest CST anchor: Lp={_n['lp']:.4f} mm, Wp={_n['wp']:.4f} mm, "
                     f"fr={_n['fr']:.4f} GHz, dist={_n['dist']:.4f} mm")
        st.download_button(
            "Download design report", _rpt.encode("utf-8"),
            file_name=f"antenna_{int(d['id'])}_{d['lp']:.1f}x{d['wp']:.1f}.txt",
            mime="text/plain", width="stretch",
        )

    with st.expander("Explore the design space (resonance vs. Lp / Wp)"):
        _ds = antenna_charts.design_space_fig(db, int(d["id"]), _shape_name)
        if _ds is not None:
            st.plotly_chart(_ds, width="stretch", config={"displayModeBar": False})
            st.caption("Read directly from the CST dataset, so this costs no model calls. "
                       "Use it to see which Lp/Wp region reaches your target frequency "
                       "before running the inverse solver.")
        else:
            st.info("Not enough CST points for this shape to map the design space.")

st.markdown("### Engine Validation")
st.caption(
    "**Leave-one-out** fidelity check. Each sampled CST point is *removed* from the lookup "
    "table before the engine is asked to predict it, so the engine has to interpolate from "
    "neighbouring geometries. That is the quantity that matters: how well it predicts a "
    "design it has never seen. (The previous version sampled rows and then queried those "
    "same rows, so the engine returned the ground truth and every error read 0.000.)"
)
col_t1, col_t2 = st.columns([1, 2])
with col_t1:
    _n_holdout = st.slider("Held-out samples", 40, 400, 220, step=20)
    if st.button("Run Self-Test (CSV fidelity)", width="stretch"):
        with st.spinner("Running leave-one-out benchmark..."):
            try:
                stats = run_engine_self_test(db, sample_size=int(_n_holdout))
                st.session_state.self_test_stats = stats
                st.session_state.self_test_ran = True
                if stats is None:
                    st.warning("Self-test completed but returned no metrics (check CSV frequency columns 1-7 GHz).")
                else:
                    st.success(f"Leave-one-out benchmark complete on {stats['n']} held-out geometries.")
            except Exception as e:
                st.session_state.self_test_stats = None
                st.session_state.self_test_ran = True
                st.error(f"Self-test failed: {e}")
with col_t2:
    if "self_test_stats" in st.session_state and st.session_state.self_test_stats is not None:
        s = st.session_state.self_test_stats
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Held-out points", f"{s['n']}")
        a2.metric("fr MAE", f"{s['fr_mae_mhz']:.1f} MHz")
        a3.metric("S11 MAE", f"{s['s11_mae_db']:.2f} dB")
        a4.metric("BW MAE", f"{s['bw_mae_mhz']:.1f} MHz")
        b1, b2, b3 = st.columns(3)
        b1.metric("fr R²", f"{s['fr']['r2']:.3f}" if s.get("fr") else "n/a")
        b2.metric("S11 R²", f"{s['s11']['r2']:.3f}" if s.get("s11") else "n/a")
        b3.metric(f"fr within ±{s['fr_tol_pct']:.0f}%", f"{s['fr_pass_pct']:.1f}%")
        st.caption("Interpolation modes used: " + ", ".join(
            f"`{k}` ×{v}" for k, v in s.get("anchor_modes", {}).items()))

        _rows = []
        for _k, _u, _lbl in [("fr", "MHz", "Resonant frequency"), ("s11", "dB", "S11 minimum"),
                             ("bw", "MHz", "Bandwidth"), ("gain", "dBi", "Gain at fr"),
                             ("eff", "%", "Radiation efficiency")]:
            _m = s.get(_k)
            if _m:
                _rows.append({"Quantity": _lbl, "Unit": _u,
                              "MAE": round(_m["mae"], 4), "P90": round(_m["p90"], 4),
                              "Max": round(_m["max"], 4), "R²": round(_m["r2"], 4),
                              "n": _m["n"]})
        if _rows:
            st.dataframe(pd.DataFrame(_rows), width="stretch", hide_index=True)

        _sc = s.get("scatter", {}).get("fr")
        if _sc:
            _pf = antenna_charts.error_scatter_fig(
                _sc[0], _sc[1], xlabel="CST ground truth (GHz)", ylabel="Engine prediction (GHz)",
                title="Resonant frequency — held-out parity", unit=" GHz")
            if _pf is not None:
                st.plotly_chart(_pf, width="stretch", config={"displayModeBar": False})
                st.caption("Points on the dashed line are exact. Spread above/below it is the "
                           "interpolation error on geometries the engine has never seen.")
    elif st.session_state.get("self_test_ran", False):
        st.info("No self-test metrics to display yet.")

st.markdown("### CST interpolation vs. the trained model")
st.caption(
    "The app ships **two** predictors. The answers you see come from interpolating the CST "
    "dataset; the trained Keras surrogate is loaded but only used as a fallback (or when you "
    "tick *Use ML refinement*). Its training-time accuracy was measured on its own held-out "
    "split, which is not the same as measuring it here. This scores both paths against the "
    "**same** held-out geometries, so the choice of default is evidence-based rather than assumed."
)
_ab1, _ab2 = st.columns([1, 2])
with _ab1:
    if fwd is None:
        st.info("No Keras model is loaded, so there is nothing to compare against.")
    elif st.button("Compare CST vs. model", width="stretch"):
        with st.spinner("Scoring both prediction paths on identical held-out geometries..."):
            try:
                st.session_state.ab_test = run_engine_ab_test(db, fwd, s_geo, s_perf, sample_size=150)
                st.session_state.ab_ran = True
            except Exception as e:
                st.session_state.ab_test = None
                st.session_state.ab_ran = True
                st.error(f"Comparison failed: {e}")
with _ab2:
    _ab = st.session_state.get("ab_test")
    if _ab:
        _cmp = [("Resonant frequency", "MHz", "fr_mhz"), ("S11 minimum", "dB", "s11_db"),
                ("Bandwidth", "MHz", "bw_mhz"), ("Gain at fr", "dBi", "gain_dbi")]
        _rows, _csv_wins, _ml_wins = [], 0, 0
        for _lbl, _u, _k in _cmp:
            _c, _m = _ab["csv"].get(_k, float("nan")), _ab["model"].get(_k, float("nan"))
            if not (np.isfinite(_c) and np.isfinite(_m)):
                continue
            if _c < _m:
                _w, _csv_wins = "CST interpolation", _csv_wins + 1
            elif _m < _c:
                _w, _ml_wins = "Trained model", _ml_wins + 1
            else:
                _w = "tie"
            _rows.append({"Quantity": _lbl, "Unit": _u,
                          "CST interpolation": round(_c, 4), "Trained model": round(_m, 4),
                          "Lower error": _w})
        if _rows:
            st.dataframe(pd.DataFrame(_rows), width="stretch", hide_index=True)
            st.caption(f"Scored on {_ab['n']} identical held-out geometries. "
                       f"Lower MAE wins. **CST { _csv_wins } — model { _ml_wins }**.")
            if _ml_wins > _csv_wins:
                st.success("The trained model is more accurate on held-out geometries. "
                           "Consider making it the default answer (Option A).")
            elif _csv_wins > _ml_wins:
                st.info("CST interpolation is more accurate on held-out geometries. "
                        "Keeping it as the default is justified (Option C) — but the UI "
                        "should say so plainly instead of implying the model produced it.")
            else:
                st.warning("Neither path is clearly better. Decide on other grounds — "
                           "generalisation far from the dataset favours the model.")
    elif st.session_state.get("ab_ran", False):
        st.info("No comparison available.")

st.markdown('<span id="designer" class="section-anchor"></span>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Inverse Design</div>', unsafe_allow_html=True)
_, mid_i, _ = st.columns([1, 2.2, 1])
with mid_i:
    tfr = st.number_input("Target Frequency (GHz)", 1.0, 7.0, 3.5, step=0.01)
    priority = st.selectbox("Optimization Priority", ["Balanced Profile", "Maximize Gain", "Maximize Bandwidth"])
    family = st.selectbox("Shape Family", list(FAMILY_MAP.keys()))
    if st.button("Synthesize Optimal Geometry"):
        with st.spinner("Search-and-optimize in progress for best physical match..."):
            best = run_inverse_solver(float(tfr), family, priority, fwd, s_geo, s_perf, db)
            if best is None:
                st.error("No valid candidate found.")
            else:
                err_pct = (float(best["err"]) / max(1e-8, float(tfr))) * 100.0
                st.session_state.inv_res = {
                    "lp": float(best["lp"]),
                    "wp": float(best["wp"]),
                    "id": int(best["shape_id"]),
                    "fr": float(best["fr"]),
                    "err_pct": float(err_pct),
                    "gain": float(best["gain"]),
                    "bw": float(best["bw"]),
                    "status": str(best["status"]),
                    "priority": priority,
                }

if "inv_res" in st.session_state:
    r = st.session_state.inv_res
    st.markdown(
        '<div class="result-banner">Inverse search completed. Review dimensions and predicted performance in this section.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<h3 class='status-heading'>Optimized Geometry</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Optimal Lp", f"{r['lp']:.3f} mm")
    c2.metric("Optimal Wp", f"{r['wp']:.3f} mm")
    c3.metric("Design Accuracy", f"{100 - r['err_pct']:.2f} %", delta=f"{r['fr']:.3f} GHz")
    c4, c5, c6 = st.columns(3)
    c4.metric("Priority", r["priority"])
    c5.metric("Pred Gain", f"{r['gain']:.2f} dBi")
    c6.metric("Pred BW", f"{r['bw']:.3f} GHz")
    st.caption(f'Status: {r["status"]}')
    _, _inv_ant_col, _ = st.columns([1.4, 1, 1.4])
    with _inv_ant_col:
        st.markdown('<div class="antenna-plot-wrap">', unsafe_allow_html=True)
        st.pyplot(draw_antenna(r["id"], round(r["lp"], 2), round(r["wp"], 2)))
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("### Inverse Validation")
i1, i2 = st.columns([1, 2])
with i1:
    inv_test_mode = st.selectbox("Inverse Test Mode", ["Quick", "Full"], index=0)
    inv_per_family = st.slider("Samples per family", min_value=2, max_value=8, value=3 if inv_test_mode == "Quick" else 5, step=1)
    if st.button("Run Inverse Self-Test"):
        fast_mode = inv_test_mode == "Quick"
        spinner_msg = "Benchmarking inverse solver (quick mode, ~10-30s)..." if fast_mode else "Benchmarking inverse solver (full mode, may take longer)..."
        with st.spinner(spinner_msg):
            try:
                t0 = time.perf_counter()
                s, fam_df = run_inverse_self_test(db, fwd, s_geo, s_perf, per_family=inv_per_family, fast_mode=fast_mode)
                elapsed = time.perf_counter() - t0
                st.session_state.inv_test_summary = s
                st.session_state.inv_test_family = fam_df
                st.session_state.inv_test_runtime_s = float(elapsed)
                st.session_state.inv_test_ran = True
                if s is None:
                    st.warning("Inverse self-test completed but no metrics were produced.")
                else:
                    st.success("Inverse self-test completed successfully.")
            except Exception as e:
                st.session_state.inv_test_summary = None
                st.session_state.inv_test_family = None
                st.session_state.inv_test_ran = True
                st.error(f"Inverse self-test failed: {e}")
with i2:
    if "inv_test_summary" in st.session_state and st.session_state.inv_test_summary:
        s = st.session_state.inv_test_summary
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tests", f"{s['tests']}")
        m2.metric("Mean fr error", f"{s['mean_err_mhz']:.1f} MHz")
        m3.metric("P90 fr error", f"{s['p90_err_mhz']:.1f} MHz")
        m4.metric("Max fr error", f"{s['max_err_mhz']:.1f} MHz")
        if "inv_test_runtime_s" in st.session_state:
            st.caption(f'Runtime: {st.session_state["inv_test_runtime_s"]:.2f} s')
        if "inv_test_family" in st.session_state and st.session_state.inv_test_family is not None:
            st.dataframe(st.session_state.inv_test_family, width="stretch")
            st.download_button(
                "Download Inverse Test Table (CSV)",
                st.session_state.inv_test_family.to_csv(index=False).encode("utf-8"),
                file_name="inverse_self_test_results.csv",
                mime="text/csv",
            )
    elif st.session_state.get("inv_test_ran", False):
        st.info("No inverse self-test metrics to display yet.")

st.markdown('<span id="assistant" class="section-anchor"></span>', unsafe_allow_html=True)
st.markdown('<div class="section-title">AETHER Assistant</div>', unsafe_allow_html=True)
_, chat_col, _ = st.columns([1, 3, 1])
with chat_col:
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key:
        try:
            groq_key = st.secrets.get("GROQ_API_KEY", "").strip()
        except Exception:
            pass
    if not groq_key:
        st.info("Set GROQ_API_KEY in `.streamlit/secrets.toml` or your environment to enable AETHER Assistant.")
    else:
        client = Groq(api_key=groq_key)
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if prompt := st.chat_input("Ask AETHER about antenna design, parameters, or results..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            response = None
            last_err = None
            # AETHER model fallback list — ordered by preference.
            # Groq occasionally deprecates model IDs; we try several so the
            # assistant keeps working without a code deploy.
            AETHER_MODELS = [
                "openai/gpt-oss-20b",          # Groq recommended replacement for llama-3.1-8b
                "qwen/qwen3.6-27b",             # fallbacks if primary is busy/down
                "openai/gpt-oss-120b",
                "llama-3.3-70b-versatile",      # older names (might still work for some tiers)
                "llama-3.1-8b-instant",
                "llama3-70b-8192",
                "llama3-8b-8192",
                "mixtral-8x7b-32768",
            ]
            # Try whichever model answered last, first. Walking an 8-entry list that
            # contains retired IDs costs one failed network round-trip per dead entry
            # on EVERY message before a live one is found.
            _preferred = st.session_state.get("aether_model")
            _ordered = ([_preferred] if _preferred else []) + [m for m in AETHER_MODELS if m != _preferred]
            try:
                for model_id in _ordered:
                    try:
                        completion = client.chat.completions.create(
                            model=model_id,
                            messages=st.session_state.messages,
                        )
                        response = completion.choices[0].message.content
                        if response:
                            st.session_state["aether_model"] = model_id
                            break
                    except Exception as model_err:
                        last_err = model_err
                        continue
            except GroqAuthError:
                st.session_state.messages.pop()
                st.error(
                    "Invalid Groq API key. In Streamlit/fly.io secrets set `GROQ_API_KEY` "
                    "to a valid key from [console.groq.com](https://console.groq.com/keys). "
                    "Use only the key (starts with `gsk_`), no quotes, then re-deploy."
                )
                response = None
            except Exception as e:
                st.session_state.messages.pop()
                st.error(f"AETHER request failed: {e}")
                response = None
            if response is None and last_err is not None:
                st.session_state.messages.pop()
                st.error(
                    "AETHER is temporarily unavailable (all Groq models were rejected — "
                    f"last error: {last_err}). Try again later or check your API key."
                )
            if response:
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown(
    """
<div class="footer-text">
Technical College of Engineering &mdash; Communication Department<br>
Sulaymaniyah Polytechnic University (SPU)<br>
<small>Graduation project &middot; Academic year 2025&ndash;2026 &middot; AI-Based CPW Antenna Synthesis</small><br>
<small style="opacity:0.75;">Submitted for academic evaluation. All engineering data derived from CST simulation datasets.</small>
</div>
""",
    unsafe_allow_html=True,
)

components.html(
    """<script>
(function () {
  var W = window.top || window.parent || window;
  function doc() {
    try { return W.document; } catch (e) { return document; }
  }
  function revealNear(anchor) {
    var node = anchor;
    for (var i = 0; i < 20 && node; i++) {
      node = node.nextElementSibling;
      if (!node) break;
      if (node.classList && (
        node.classList.contains("hero-container") ||
          node.classList.contains("hero-content") ||
          node.classList.contains("intro-panel") ||
          node.classList.contains("section-title")
      )) {
        node.classList.add("section-reveal");
        setTimeout(function (n) {
          return function () { n.classList.remove("section-reveal"); };
        }(node), 900);
        return;
      }
    }
  }
  function scrollToId(sectionId, clickedLink) {
    var d = doc();
    var target = d.getElementById(sectionId);
    if (!target) return false;
    target.scrollIntoView({ behavior: "smooth", block: "start" });
    var navItems = d.querySelectorAll(".nav-item");
    navItems.forEach(function (l) { l.classList.remove("active"); });
    if (clickedLink && clickedLink.classList.contains("nav-item")) {
      clickedLink.classList.add("active");
      clickedLink.classList.add("is-scrolling");
      setTimeout(function () { clickedLink.classList.remove("is-scrolling"); }, 600);
    }
    revealNear(target);
    return true;
  }
  function bindNav() {
    var d = doc();
    if (!d.querySelector(".nav-item")) return false;
    if (W.__spuNavClickBound) return true;
    W.__spuNavClickBound = true;
    d.addEventListener("click", function (e) {
      var link = e.target.closest(".nav-item, .nav-home-link");
      if (!link) return;
      var href = link.getAttribute("href");
      if (!href || href.charAt(0) !== "#") return;
      e.preventDefault();
      scrollToId(href.slice(1), link);
    }, true);
    var ids = ["overview", "introduction", "intel", "simulator", "designer", "assistant"];
    var sections = ids.map(function (id) { return d.getElementById(id); }).filter(Boolean);
    function onScroll() {
      var current = 0;
      sections.forEach(function (sec, i) {
        if (sec.getBoundingClientRect().top <= 130) current = i;
      });
      d.querySelectorAll(".nav-item").forEach(function (l, i) {
        l.classList.toggle("active", i === current);
      });
    }
    W.addEventListener("scroll", onScroll, { passive: true });
    d.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return true;
  }
  var tries = 0;
  var timer = setInterval(function () {
    tries += 1;
    if (bindNav() || tries > 40) clearInterval(timer);
  }, 350);
})();
</script>""",
    height=0,
)