"""
Two-Brain 2.0 - Inference engine
================================
Loads:
  - forward_model_v2_curve.keras   (main brain, dip-weighted, 2 heads)
  - s11_dip_specialist.pkl        (specialist: predicts s11_min and f_dip)
  - scaler_geo.pkl, scaler_perf.pkl

Public API:
  load_v2_engines()           -> (main_model, dip_specialist_dict, s_geo, s_perf, FREQS_RAW)
  forward_v2(...)             -> dict with fr, s11, bw, gain, eff, status, f, c, v, ...
  predict_dip_specialist(...) -> (f_dip_ghz, s11_min_db)
  apply_anchor_blend(...)     -> the 50-MHz window replacement of the curve

Backward compatible with 10_web_app.py v1 forward_ml() / forward_anchor() helpers.
"""
import os
import numpy as np
import joblib
import tensorflow as tf

# Respect the HF_REPO environment variable (fly.toml / Dockerfile set it). It was
# previously hardcoded to a different account's repo than 10_web_app.py used, so the
# dip specialist was never fetched from the right place.
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


FREQS_RAW = np.linspace(1.0, 7.0, 151)


def load_v2_engines(model_dir=None, main_model=None):
    """Load the Two-Brain 2.0 stack (or fall back to v1 if v2 is missing).

    main_model : optional already-loaded Keras model. 10_web_app.load_brains() loads
      forward_model_v2_curve.keras for the v1 path; passing that same object here
      avoids loading the identical 115 MB model a second time. Holding two full
      copies of it plus TensorFlow inside a 1 GB container was the single largest
      memory cost in the app and the most likely cause of OOM restarts.
    """
    if model_dir is not None:
        os.chdir(model_dir)

    if main_model is not None:
        return _finish_load(main_model)

    main_path = "forward_model_v2_curve.keras"
    if not os.path.exists(main_path):
        _download_hf("forward_model_v2_curve.keras")
    if not os.path.exists(main_path):
        for alt in ("forward_model_shrunk.keras", "forward_model_final.keras", "forward_model_final.h5"):
            if os.path.exists(alt):
                main_path = alt
                break
        else:
            raise FileNotFoundError(
                "No main brain found. Expected one of: "
                "forward_model_v2_curve.keras, forward_model_shrunk.keras, "
                "forward_model_final.keras, forward_model_final.h5"
            )

    main = tf.keras.models.load_model(main_path, compile=False)
    return _finish_load(main)


def _finish_load(main):
    """Shared tail of load_v2_engines(): attach the specialist + scalers."""
    dip = None
    _download_hf("s11_dip_specialist.pkl")
    if os.path.exists("s11_dip_specialist.pkl"):
        try:
            dip = joblib.load("s11_dip_specialist.pkl")
        except Exception as e:
            print(f"[HF] Could not load specialist: {e}")
            dip = None
    if dip is None:
        # Surface this clearly: without the specialist the app runs the CSV anchor
        # path only, and the "Two-Brain 2.0" label would be misleading.
        print("[ENGINE] s11_dip_specialist.pkl unavailable -> running main brain only "
              f"(looked in HF repo '{HF_REPO}')")

    s_geo = joblib.load("scaler_geo.pkl")
    s_perf = joblib.load("scaler_perf.pkl")

    return main, dip, s_geo, s_perf, FREQS_RAW


# ---------------------------------------------------------------------------
# DSP / VNA helpers
# ---------------------------------------------------------------------------
def _apply_dsp_polishing(freqs, curve):
    from scipy.interpolate import make_interp_spline

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

    # Cubic spline upsampling from 151 to 1000 pts. NO Savitzky-Golay — it destroys dip accuracy.
    f_fine = np.linspace(freqs.min(), freqs.max(), 1000)
    k = min(3, len(freqs) - 1)
    spline = make_interp_spline(freqs, curve, k=k)
    return f_fine, spline(f_fine)


def _vna_measure(freqs, curve):
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


# ---------------------------------------------------------------------------
# Dip specialist
# ---------------------------------------------------------------------------
def predict_dip_specialist(x_geo_scaled, dip_specialist):
    """x_geo_scaled: (1, 14) already transformed. Returns (f_dip_ghz, s11_min_db)."""
    if dip_specialist is None:
        return None, None

    s11_w = dip_specialist["s11_weights"]
    s11_pred = (
        s11_w[0] * dip_specialist["gbr_s11"].predict(x_geo_scaled)
        + s11_w[1] * dip_specialist["rf_s11"].predict(x_geo_scaled)
        + s11_w[2] * dip_specialist["et_s11"].predict(x_geo_scaled)
    )

    fdip_w = dip_specialist["fdip_weights"]
    x_aug = np.hstack([x_geo_scaled, s11_pred.reshape(-1, 1)])
    fdip_pred = (
        fdip_w[0] * dip_specialist["gbr_fdip"].predict(x_aug)
        + fdip_w[1] * dip_specialist["rf_fdip"].predict(x_aug)
    )
    return float(fdip_pred[0]), float(s11_pred[0])


# ---------------------------------------------------------------------------
# Anchor-blend in a window around (f_dip, s11_min)
# ---------------------------------------------------------------------------
def apply_anchor_blend(freqs, curve, f_dip, s11_min, window_mhz=50.0, anchor_strength=1.0):
    """
    Replace the main brain's curve in a +/- window_mhz/2 around f_dip with a
    smooth sine-envelope interpolation that:
      - has the specialist's (f_dip, s11_min) point exactly in the middle
      - meets the main brain's curve at the window edges with zero slope
      - leaves the rest of the curve untouched

    anchor_strength in [0, 1]:
        1.0 = full specialist override
        0.5 = 50/50 blend
        0.0 = no change
    """
    f = np.asarray(freqs, dtype=float).copy()
    c = np.asarray(curve, dtype=float).copy()

    half = window_mhz / 2000.0  # MHz -> GHz
    in_win = (f >= f_dip - half) & (f <= f_dip + half)
    if not np.any(in_win) or anchor_strength <= 0.0:
        return c

    idx_in = np.where(in_win)[0]
    i_lo, i_hi = idx_in[0], idx_in[-1]
    c_lo, c_hi = c[i_lo], c[i_hi]
    f_lo, f_hi = f[i_lo], f[i_hi]

    # Smooth sine envelope:
    #   - t = 0 at f_lo, t = 1 at f_hi
    #   - ramp = sin(pi*t)  -> 0 at edges, 1 at f_dip (by symmetry of the window)
    #   - target = (1-ramp) * (c_lo + c_hi)/2  +  ramp * s11_min
    # This guarantees:
    #   - target(f_lo) = (c_lo + c_hi)/2
    #   - target(f_dip) = s11_min
    #   - target(f_hi) = (c_lo + c_hi)/2
    # which is the natural "anchor pull-down" shape.  We then blend with the
    # original curve by `anchor_strength`.
    t = (f - f_lo) / max(1e-12, (f_hi - f_lo))
    ramp = np.sin(np.pi * t)
    target = (1.0 - ramp) * 0.5 * (c_lo + c_hi) + ramp * s11_min

    blended = (1.0 - anchor_strength) * c + anchor_strength * target
    c_new = c.copy()
    c_new[in_win] = blended[in_win]
    return c_new


# ---------------------------------------------------------------------------
# Main entry point used by 10_web_app.py
# ---------------------------------------------------------------------------
def forward_v2(lp, wp, shape_id, main_model, dip_specialist, s_geo, s_perf,
               use_anchor=True, blend_window_mhz=50.0, anchor_strength=1.0,
               verbose=0):
    """
    Args:
        lp, wp    : geometry (mm)
        shape_id  : 1..12
        main_model: the TF model (v2 preferred, v1 also works)
        dip_specialist: dict or None (if None, falls back to plain main brain)
        s_geo, s_perf: scalers from 2_preprocessing.py
        use_anchor: if True and a specialist exists, blend the curve near the dip
        blend_window_mhz: width of the specialist blend window (default 50 MHz)
        anchor_strength: 0..1, how strongly the specialist overrides (default 1.0)

    Returns: dict (matches 10_web_app.py forward_ml() output format)
    """
    # Build the 14-dim input (Lp, Wp, 12 shape one-hots) and scale
    x = np.zeros((1, 14), dtype=float)
    x[0, 0] = float(lp)
    x[0, 1] = float(wp)
    if 1 <= int(shape_id) <= 12:
        x[0, 1 + int(shape_id)] = 1.0
    x_scaled = s_geo.transform(x)

    # Run main brain (2 heads: scalars [fr, Gain, Eff, S11_min], curve [151])
    p_scalars, p_curve = main_model.predict(x_scaled, verbose=verbose)
    p_scalars = p_scalars[0]
    p_curve = p_curve[0]

    # Reconstruct the full 157-dim prediction.  We put the main brain's outputs
    # in their canonical scaler_perf slots: [0]=fr, [2]=Gain, [3]=Eff, [4]=S11_min.
    # The other slots (BW=1, is_matched=5) get zeros - they're not predicted
    # by the v2 main brain.
    recon = np.zeros((1, 157), dtype=float)
    recon[0, [0, 2, 3, 4]] = p_scalars
    recon[0, 6:] = p_curve
    perf = s_perf.inverse_transform(recon)[0]
    base_s11_min = float(perf[4])  # S11_min (dB), from main brain
    base_gain = float(perf[2])
    base_eff = float(perf[3])

    # The 151-point curve in dB
    base_curve = perf[6:].astype(float)

    # Specialist dip prediction
    spec_f_dip, spec_s11_min = predict_dip_specialist(x_scaled, dip_specialist)
    if use_anchor and dip_specialist is not None and spec_f_dip is not None:
        # Clamp f_dip to the curve's frequency range
        spec_f_dip = float(np.clip(spec_f_dip, FREQS_RAW[0], FREQS_RAW[-1]))
        # Don't allow impossibly shallow dips
        spec_s11_min = float(min(spec_s11_min, -2.0))
        final_curve = apply_anchor_blend(
            FREQS_RAW, base_curve, spec_f_dip, spec_s11_min,
            window_mhz=blend_window_mhz, anchor_strength=anchor_strength,
        )
        used_specialist = True
    else:
        final_curve = base_curve
        spec_f_dip = None
        spec_s11_min = None
        used_specialist = False

    # DSP polishing + VNA measurement
    f_s, c_s = _apply_dsp_polishing(FREQS_RAW, final_curve)
    fr, s11, bw, status, vswr = _vna_measure(f_s, c_s)

    return {
        "fr": fr,
        "s11": s11,
        "bw": bw,
        "gain": base_gain,
        "eff": base_eff,
        "status": status,
        "f": f_s,
        "c": c_s,
        "v": vswr,
        # Two-Brain 2.0 extras (used for the metrics panel in the web app)
        "spec_f_dip": spec_f_dip,
        "spec_s11_min": spec_s11_min,
        "main_s11_min": base_s11_min,
        "used_specialist": used_specialist,
        "engine": "Two-Brain 2.0" if used_specialist else "Main-Brain only",
    }


def forward_v2_with_csv_anchor(lp, wp, shape_id, main_model, dip_specialist, s_geo, s_perf,
                                csv_curve, csv_f_axis,
                                use_anchor=True, blend_window_mhz=50.0, anchor_strength=1.0,
                                verbose=0):
    """
    Two-Brain 2.0 with CSV anchoring: uses CSV data as primary, specialist as dip enhancer.
    This is the correct approach - CSV data is ground truth, specialist improves dip quality.
    """
    # If we have CSV data, use it directly (CSV is ground truth)
    if csv_curve is not None and len(csv_curve) > 0:
        # Use CSV curve as the base (this is the CST simulation data)
        base_curve = csv_curve
        # Run specialist to predict where the dip SHOULD be
        x = np.zeros((1, 14), dtype=float)
        x[0, 0] = float(lp)
        x[0, 1] = float(wp)
        if 1 <= int(shape_id) <= 12:
            x[0, 1 + int(shape_id)] = 1.0
        x_scaled = s_geo.transform(x)
        
        spec_f_dip, spec_s11_min = predict_dip_specialist(x_scaled, dip_specialist)
        
        if use_anchor and dip_specialist is not None and spec_f_dip is not None:
            spec_f_dip = float(np.clip(spec_f_dip, FREQS_RAW[0], FREQS_RAW[-1]))
            spec_s11_min = float(min(spec_s11_min, -2.0))
            final_curve = apply_anchor_blend(
                csv_f_axis, base_curve, spec_f_dip, spec_s11_min,
                window_mhz=blend_window_mhz, anchor_strength=anchor_strength,
            )
            used_specialist = True
        else:
            final_curve = base_curve
            used_specialist = False
        
        f_s, c_s = _apply_dsp_polishing(csv_f_axis, final_curve)
    else:
        # No CSV data - use pure ML prediction
        result = forward_v2(lp, wp, shape_id, main_model, dip_specialist, s_geo, s_perf,
                           use_anchor=use_anchor, blend_window_mhz=blend_window_mhz,
                           anchor_strength=anchor_strength, verbose=verbose)
        return result
    
    fr, s11, bw, status, vswr = _vna_measure(f_s, c_s)
    
    return {
        "fr": fr,
        "s11": s11,
        "bw": bw,
        "gain": float(np.nan),
        "eff": float(np.nan),
        "status": status,
        "f": f_s,
        "c": c_s,
        "v": vswr,
        "spec_f_dip": spec_f_dip if used_specialist else None,
        "spec_s11_min": spec_s11_min if used_specialist else None,
        "used_specialist": used_specialist,
        "engine": "Two-Brain 2.0 + CSV" if used_specialist else "CSV-only",
    }
