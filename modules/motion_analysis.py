"""Exploratory acceleration analysis for tremor-like motion and gait samples."""
import numpy as np
import pandas as pd
from scipy.signal import detrend, find_peaks
from .confidence import confidence_report

AXIS_OPTIONS = [
    ("ax", "ay", "az"),
    ("x", "y", "z"),
    ("acceleration_x", "acceleration_y", "acceleration_z"),
]
TIME_OPTIONS = ("timestamp", "time", "t", "timestamp_seconds")

def _read_axes(df):
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    axes = next((triple for triple in AXIS_OPTIONS if all(k in lower_map for k in triple)), None)
    if axes is None:
        raise ValueError("Missing acceleration columns. Use ax, ay, az (or x, y, z).")
    xyz = df[[lower_map[k] for k in axes]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(xyz).all():
        raise ValueError("Acceleration columns contain missing or non-numeric samples.")
    time_key = next((lower_map[k] for k in TIME_OPTIONS if k in lower_map), None)
    t = None
    if time_key is not None:
        t = pd.to_numeric(df[time_key], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError("Timestamps must be finite and strictly increasing, in seconds.")
    return xyz, t

def _sampling_rate(t, n):
    if t is None:
        return 50.0, None
    diffs = np.diff(t)
    dt = float(np.median(diffs))
    if dt <= 0:
        raise ValueError("Timestamps must increase.")
    return 1.0 / dt, float(np.std(diffs) / dt) if len(diffs) > 1 else 0.0

def analyze_motion(df: pd.DataFrame) -> dict:
    if df is None or len(df) < 32:
        raise ValueError("At least 32 sensor samples are required.")
    xyz, t = _read_axes(df)
    fs, jitter = _sampling_rate(t, len(xyz))
    if fs < 8 or fs > 1000:
        raise ValueError("Estimated sample rate is outside the supported 8–1000 Hz range. Check timestamp units.")
    magnitude = np.linalg.norm(xyz, axis=1)
    signal = detrend(magnitude)
    duration = len(signal) / fs
    window = np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(signal * window)) / max(np.sum(window) / 2, 1e-12)
    frequencies = np.fft.rfftfreq(len(signal), d=1/fs)
    valid = frequencies > 0
    dominant = float(frequencies[valid][np.argmax(spectrum[valid])]) if np.any(valid) else None
    reasons = []
    passed = 0
    total = 4
    if duration >= 5: passed += 1
    else: reasons.append("Recording is shorter than 5 seconds.")
    if jitter is None or jitter <= 0.15: passed += 1
    else: reasons.append("Sensor timestamps are irregular.")
    if float(np.std(signal)) >= 1e-4: passed += 1
    else: reasons.append("Very little movement variation measured; verify sensor and placement.")
    if len(signal) >= 100: passed += 1
    else: reasons.append("Sample count is low; frequency estimates may be unstable.")
    quality = confidence_report(reasons, passed, total)
    return {
        "sample_count": int(len(signal)), "duration_seconds": float(duration),
        "sample_rate_hz": float(fs), "timing_jitter_ratio": jitter,
        "dominant_frequency_hz": dominant, "frequencies_hz": frequencies.astype(float).tolist(),
        "spectrum": spectrum.astype(float).tolist(), "magnitude": magnitude.astype(float).tolist(),
        "quality": quality, "assumed_sample_rate": t is None,
    }

def analyze_gait(df: pd.DataFrame) -> dict:
    """Very preliminary step-event heuristic; requires review, not clinical interpretation."""
    if df is None or len(df) < 100:
        raise ValueError("Gait analysis requires at least 100 samples; use a 10–20 second walk recording.")
    xyz, t = _read_axes(df)
    fs, jitter = _sampling_rate(t, len(xyz))
    if fs < 10 or fs > 200:
        raise ValueError("For this gait demo, sample rate must be 10–200 Hz. Check timestamp units.")
    magnitude = np.linalg.norm(xyz, axis=1)
    dynamic = detrend(magnitude)
    # Smooth over approximately 100 ms to reduce single-sample noise.
    smooth_n = max(1, int(fs * 0.10))
    smooth = np.convolve(dynamic, np.ones(smooth_n)/smooth_n, mode="same")
    min_distance = max(1, int(fs * 0.30))
    prominence = max(float(np.std(smooth) * 0.25), 1e-5)
    peaks, props = find_peaks(smooth, distance=min_distance, prominence=prominence)
    peak_times = peaks / fs if t is None else t[peaks] - t[0]
    intervals = np.diff(peak_times)
    plausible = intervals[(intervals >= 0.3) & (intervals <= 2.0)]
    cadence = float(60 / np.median(plausible)) if plausible.size else None
    variability = float(np.std(plausible) / np.mean(plausible) * 100) if plausible.size > 1 and np.mean(plausible) > 0 else None
    reasons = []
    passed = 0
    total = 4
    duration = len(xyz) / fs
    if duration >= 10: passed += 1
    else: reasons.append("Walk recording is shorter than 10 seconds.")
    if jitter is None or jitter <= 0.15: passed += 1
    else: reasons.append("Sensor timestamps are irregular.")
    if len(peaks) >= 6: passed += 1
    else: reasons.append("Few candidate step peaks found; placement or recording length may be unsuitable.")
    if np.std(dynamic) > 1e-3: passed += 1
    else: reasons.append("Low movement variation; confirm the phone is secured and recording.")
    return {
        "sample_count": int(len(xyz)), "duration_seconds": float(duration),
        "sample_rate_hz": float(fs), "candidate_step_events": int(len(peaks)),
        "cadence_steps_per_min": cadence, "step_interval_variability_percent": variability,
        "step_intervals_seconds": plausible.astype(float).tolist(),
        "quality": confidence_report(reasons, passed, total),
        "assumed_sample_rate": t is None,
        "method_note": "Candidate peaks from phone acceleration magnitude; not a validated step detector.",
    }
