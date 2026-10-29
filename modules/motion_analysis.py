import numpy as np
import pandas as pd
from scipy.signal import detrend, find_peaks


def analyze_motion(df: pd.DataFrame) -> dict:
    """Compute exploratory frequency features from CSV acceleration columns.

    Accepted acceleration column names: ax/ay/az, x/y/z, or acceleration_x/y/z.
    An optional timestamp column can be named timestamp, time, or t (seconds).
    """
    if df is None or len(df) < 32:
        raise ValueError("At least 32 sensor samples are required for this exploratory analysis.")

    lower_map = {str(c).strip().lower(): c for c in df.columns}
    options = [
        ("ax", "ay", "az"),
        ("x", "y", "z"),
        ("acceleration_x", "acceleration_y", "acceleration_z"),
    ]
    axes = next((triple for triple in options if all(k in lower_map for k in triple)), None)
    if axes is None:
        raise ValueError("Could not find acceleration columns. Use ax, ay, az (or x, y, z).")

    xyz = df[[lower_map[k] for k in axes]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(xyz).all():
        raise ValueError("Acceleration columns contain missing or non-numeric samples.")

    time_key = next((lower_map[k] for k in ("timestamp", "time", "t") if k in lower_map), None)
    if time_key is not None:
        t = pd.to_numeric(df[time_key], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError("Timestamps must be finite and strictly increasing.")
        dt = float(np.median(np.diff(t)))
        fs = 1.0 / dt
        timing_jitter = float(np.std(np.diff(t)) / dt) if len(t) > 2 else 0.0
    else:
        fs = 50.0  # Explicit assumption if no timestamps are provided.
        timing_jitter = None

    if fs < 8:
        raise ValueError("Estimated sample rate is too low for the exploratory frequency analysis.")
    if fs > 1000:
        raise ValueError("Estimated sample rate is unexpectedly high; check timestamp units.")

    # Vector magnitude is useful for a simple demo, but phone orientation and gravity matter.
    magnitude = np.linalg.norm(xyz, axis=1)
    signal = detrend(magnitude)
    window = np.hanning(len(signal))
    spectrum_complex = np.fft.rfft(signal * window)
    frequencies = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    spectrum = np.abs(spectrum_complex) / max(np.sum(window) / 2.0, 1e-12)

    valid = frequencies > 0
    if not np.any(valid):
        dominant = None
    else:
        idx = int(np.argmax(spectrum[valid]))
        dominant = float(frequencies[valid][idx])

    reasons = []
    duration = len(signal) / fs
    if duration < 5:
        reasons.append("Recording is shorter than five seconds.")
    if timing_jitter is not None and timing_jitter > 0.15:
        reasons.append("Sensor timestamps are irregular.")
    if float(np.std(signal)) < 1e-4:
        reasons.append("Very little variation was measured; verify the sensor and placement.")
    status = "usable" if not reasons else "repeat_recommended"
    message = "Basic signal checks passed; this is not a clinical interpretation." if not reasons else "Review the sensor recording: " + " ".join(reasons)

    return {
        "sample_count": int(len(signal)),
        "duration_seconds": float(duration),
        "sample_rate_hz": float(fs),
        "timing_jitter_ratio": timing_jitter,
        "dominant_frequency_hz": dominant,
        "frequencies_hz": frequencies.astype(float).tolist(),
        "spectrum": spectrum.astype(float).tolist(),
        "quality": {"status": status, "message": message, "reasons": reasons},
    }
