from io import BytesIO
import numpy as np
import soundfile as sf
import librosa


def analyze_audio_bytes(audio_bytes: bytes) -> dict:
    """Analyze an in-memory WAV payload. No files are written by this function."""
    if not audio_bytes:
        raise ValueError("The uploaded file is empty.")

    audio, sample_rate = sf.read(BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)

    if sample_rate <= 0 or audio.size < sample_rate:
        raise ValueError("Please provide at least one second of valid audio.")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains invalid numeric values.")

    duration = float(len(audio) / sample_rate)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.999))
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))

    # A robust pitch estimate is optional; some samples may not contain reliable voiced speech.
    pitch_median = None
    try:
        f0 = librosa.yin(audio, fmin=65, fmax=400, sr=sample_rate)
        voiced = f0[np.isfinite(f0) & (f0 >= 65) & (f0 <= 400)]
        if voiced.size:
            pitch_median = float(np.median(voiced))
    except Exception:
        pitch_median = None

    reasons = []
    if duration < 5:
        reasons.append("Recording is very short for a speech assessment.")
    if rms < 0.003:
        reasons.append("Audio level is very low.")
    if clipping_ratio > 0.01:
        reasons.append("Possible clipping detected.")
    status = "usable" if not reasons else "repeat_recommended"
    message = "Basic signal checks passed; this does not establish clinical validity." if not reasons else "Review the recording: " + " ".join(reasons)

    return {
        "duration_seconds": duration,
        "sample_rate": int(sample_rate),
        "peak_amplitude": peak,
        "rms": rms,
        "clipping_ratio": clipping_ratio,
        "mfcc_means": np.mean(mfcc, axis=1).astype(float).tolist(),
        "mfcc_stds": np.std(mfcc, axis=1).astype(float).tolist(),
        "zero_crossing_rate": zcr,
        "pitch_median_hz": pitch_median,
        "quality": {"status": status, "message": message, "reasons": reasons},
    }
