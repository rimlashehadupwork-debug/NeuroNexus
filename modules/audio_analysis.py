"""In-memory acoustic feature extraction for research demonstrations."""
from io import BytesIO
import numpy as np
import soundfile as sf
import librosa
from .confidence import confidence_report


def to_standard_wav(audio_bytes: bytes, target_sr: int = 16000) -> tuple[bytes, dict]:
    """Decode any common audio format (WAV/MP3/M4A/OGG/FLAC) to 16 kHz mono WAV.

    Returns (wav_bytes, info). `info` reports what conversion happened so the
    UI can tell the user their file was converted rather than rejected.
    """
    if not audio_bytes:
        raise ValueError("The audio input is empty.")
    audio, sr, decoded_with = None, None, None
    try:  # fast path: formats libsndfile reads directly (wav, flac, ogg)
        audio, sr = sf.read(BytesIO(audio_bytes), dtype="float32", always_2d=False)
        decoded_with = "soundfile"
    except Exception:
        try:  # fallback: librosa/audioread path (mp3, m4a, ...)
            audio, sr = librosa.load(BytesIO(audio_bytes), sr=None, mono=True)
            decoded_with = "librosa"
        except Exception as exc:
            raise ValueError(
                "Could not decode this audio file. Supported: WAV, MP3, M4A, OGG, FLAC."
            ) from exc
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains missing or invalid numeric values.")
    source_sr, source_len = int(sr), int(audio.size)
    if source_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=source_sr, target_sr=target_sr)
    buf = BytesIO()
    sf.write(buf, audio, target_sr, format="WAV", subtype="PCM_16")
    info = {
        "decoded_with": decoded_with,
        "source_sample_rate": source_sr,
        "target_sample_rate": target_sr,
        "source_seconds": round(source_len / float(source_sr), 2) if source_sr else 0.0,
        "converted": source_sr != target_sr or decoded_with != "soundfile",
    }
    return buf.getvalue(), info


def analyze_audio_bytes(audio_bytes: bytes) -> dict:
    if not audio_bytes:
        raise ValueError("The audio input is empty.")
    try:
        audio, sample_rate = sf.read(BytesIO(audio_bytes), dtype="float32", always_2d=False)
    except Exception as exc:
        raise ValueError("Could not decode this audio. Please provide a valid WAV file.") from exc
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate <= 0 or audio.size < sample_rate:
        raise ValueError("Please provide at least one second of valid audio.")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains missing or invalid numeric values.")
    duration = float(len(audio) / sample_rate)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.999))
    reasons = []
    checks_total = 5
    passed = 0
    if duration >= 5: passed += 1
    else: reasons.append("Recording is shorter than 5 seconds; repeat with a longer sample.")
    if rms >= 0.003: passed += 1
    else: reasons.append("Audio level is very low; move closer to the microphone.")
    if clipping_ratio <= 0.01: passed += 1
    else: reasons.append("Possible clipping detected; reduce input volume.")
    if sample_rate >= 16000: passed += 1
    else: reasons.append("Sample rate is below 16 kHz; recording detail may be limited.")
    # A simple silence ratio helps flag recordings dominated by near-silence.
    frame_len = max(1, int(sample_rate * 0.025))
    frame_rms = np.array([
        np.sqrt(np.mean(audio[i:i+frame_len]**2))
        for i in range(0, len(audio), frame_len) if len(audio[i:i+frame_len])
    ])
    silence_ratio = float(np.mean(frame_rms < 0.001))
    if silence_ratio < 0.8: passed += 1
    else: reasons.append("Most frames are near-silent; check microphone placement and repeat.")
    # Feature extraction is bounded and runs in memory.
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=int(sample_rate), n_mfcc=13)
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=int(sample_rate))))
        rms_frames = librosa.feature.rms(y=audio)[0]
        rms_variability = float(np.std(rms_frames))
    except Exception as exc:
        raise ValueError("Audio feature extraction failed. Try a clean mono WAV file.") from exc
    pitch_median = None
    try:
        f0 = librosa.yin(audio, fmin=65, fmax=min(400, sample_rate / 2 - 1), sr=int(sample_rate))
        voiced = f0[np.isfinite(f0) & (f0 >= 65) & (f0 <= 400)]
        if voiced.size:
            pitch_median = float(np.median(voiced))
    except Exception:
        pass
    confidence = confidence_report(reasons, passed, checks_total)
    return {
        "duration_seconds": duration, "sample_rate": int(sample_rate),
        "peak_amplitude": peak, "rms": rms, "clipping_ratio": clipping_ratio,
        "silence_ratio": silence_ratio,
        "mfcc_means": np.mean(mfcc, axis=1).astype(float).tolist(),
        "mfcc_stds": np.std(mfcc, axis=1).astype(float).tolist(),
        "zero_crossing_rate": zcr, "spectral_centroid_hz": spectral_centroid,
        "rms_variability": rms_variability, "pitch_median_hz": pitch_median,
        "quality": confidence,
    }
