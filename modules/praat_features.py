"""Experimental: analyze a user's voice recording with the Parkinson's research model.

Pipeline: any recorded/uploaded audio -> WAV 16kHz mono -> Praat-compatible
acoustic features (jitter, shimmer, pitch statistics, voice breaks) ->
research model -> EXPERIMENTAL pattern output.

HONESTY RULES (from the NeuroNexus research plan):
- Output is shown ONLY if recording-quality gates pass AND the task was a
  sustained vowel (the research dataset contains phonations, not read speech).
- The output is a research pattern comparison, never a diagnosis or a disease
  probability. Subject-level balanced accuracy is ~0.55-0.62 with a wide CI.
- Dataset features were computed with specific Praat settings; phone
  microphones and different settings shift feature distributions, which the
  UI must state next to any output.
"""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
import json
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

try:
    import parselmouth
except ImportError:
    parselmouth = None

TARGET_SR = 16000
MIN_VOICED_SECONDS = 3.0


def to_wav_mono16k(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode any supported audio format to mono 16 kHz float waveform."""
    audio, sr = sf.read(BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = (audio / peak * 0.9).astype(np.float32)  # gentle normalization
    return audio, TARGET_SR


def _praat_features(wav: np.ndarray, sr: int) -> dict:
    """Compute the 26 dataset-compatible features using Praat (parselmouth)."""
    if parselmouth is None:
        raise RuntimeError(
            "Praat pattern analysis requires praat-parselmouth. "
            "Install the project requirements and restart the app."
        )
    try:
        from parselmouth.praat import call
    except ImportError as exc:
        raise RuntimeError(
            "Praat pattern analysis is unavailable because praat-parselmouth "
            "could not be loaded."
        ) from exc

    snd = parselmouth.Sound(wav, sampling_frequency=sr)
    pitch = call(snd, "To Pitch", 0.0, 75.0, 600.0)
    point = call(snd, "To PointProcess (periodic, cc)", 75.0, 600.0)
    d = snd.duration
    f0 = pitch.selected_array["frequency"]
    f0v = f0[f0 > 0]
    if f0v.size < sr * MIN_VOICED_SECONDS / 100.0:  # too few voiced frames
        raise ValueError("Too little voiced audio. Sustain a steady vowel such as 'ah' for at least 5 seconds.")
    ac = call([snd, point], "Get maximum autocorrelation", 0.0, d, 75.0, 600.0, 1.3)
    hnr = call(snd, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)
    hnr_db = call(hnr, "Get mean", 0.0, d)
    rap = call(point, "Get jitter (rap)", 0.0, d, 0.0001, 0.02, 1.3)
    apq3 = call([snd, point], "Get shimmer (apq3)", 0.0, d, 0.0001, 0.02, 1.3, 1.6)
    return {
        "jitter_local": call(point, "Get jitter (local)", 0.0, d, 0.0001, 0.02, 1.3),
        "jitter_local_abs": call(point, "Get jitter (local, absolute)", 0.0, d, 0.0001, 0.02, 1.3),
        "jitter_rap": rap,
        "jitter_ppq5": call(point, "Get jitter (ppq5)", 0.0, d, 0.0001, 0.02, 1.3),
        "jitter_ddp": 3.0 * rap,
        "shimmer_local": call([snd, point], "Get shimmer (local)", 0.0, d, 0.0001, 0.02, 1.3, 1.6),
        "shimmer_local_db": call([snd, point], "Get shimmer (local_dB)", 0.0, d, 0.0001, 0.02, 1.3, 1.6),
        "shimmer_apq3": apq3,
        "shimmer_apq5": call([snd, point], "Get shimmer (apq5)", 0.0, d, 0.0001, 0.02, 1.3, 1.6),
        "shimmer_apq11": call([snd, point], "Get shimmer (apq11)", 0.0, d, 0.0001, 0.02, 1.3, 1.6),
        "shimmer_dda": 3.0 * apq3,
        "ac": ac,
        "nth": 1.0 / (10 ** (hnr_db / 10.0) + 1.0),  # noise-to-harmonics derived from HNR
        "htn": hnr_db,
        "median_pitch": float(np.median(f0v)),
        "mean_pitch": float(np.mean(f0v)),
        "std_pitch": float(np.std(f0v)),
        "min_pitch": float(np.min(f0v)),
        "max_pitch": float(np.max(f0v)),
        "num_pulses": call(point, "Get number of points", 0.0, d),
        "num_periods": call(point, "Get number of periods", 0.0, d, 0.0001, 0.02, 1.3),
        "mean_period": call(point, "Get mean period", 0.0, d, 0.0001, 0.02, 1.3),
        "std_period": call(point, "Get stdev of periods", 0.0, d, 0.0001, 0.02, 1.3),
        "frac_unvoiced": float(np.mean(f0 <= 0)),
        "num_voice_breaks": call(point, "Count voice breaks", 0.0, d),
        "degree_voice_breaks": call(point, "Get degree of voice breaks", 0.0, d, 0.0001, 0.02, 1.3),
    }


def extract_voice_features(audio_bytes: bytes) -> tuple[np.ndarray, dict]:
    """WAV conversion + 26-feature extraction. Returns (wav, features)."""
    wav, sr = to_wav_mono16k(audio_bytes)
    return wav, _praat_features(wav, sr)


def run_symptom_pattern_analysis(audio_bytes: bytes, models_dir: str | Path = "models") -> dict:
    """Full gated pipeline. Returns a dict safe to render in the UI.

    Gates: decodable WAV, >=5 s, passes the app's quality checks, enough voiced
    audio. The model output is labeled experimental and carries its metrics
    and limitations so the UI never presents it as a diagnosis.
    """
    from .audio_analysis import analyze_audio_bytes  # existing quality checks
    from .model_registry import load_parkinson_model

    quality = analyze_audio_bytes(audio_bytes)  # raises if undecodable/too short
    wav, feats = extract_voice_features(audio_bytes)

    pipeline, feature_columns, threshold = load_parkinson_model(models_dir)
    row = pd.DataFrame([feats], columns=None)[feature_columns]
    prob = float(pipeline.predict_proba(row)[0, 1])
    pred = int(prob >= threshold)

    meta_path = Path(models_dir) / "parkinson_voice_research_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return {
        "compatible_features": feats,
        "recording_quality": quality["quality"],
        "experimental_output": {
            "pattern_label": "patterns closer to dataset controls" if pred == 0 else "patterns closer to dataset Parkinson's-labelled recordings",
            "model_score": round(prob, 3),
            "threshold": threshold,
        },
        "model_context": {
            "model": meta.get("model_name"),
            "version": meta.get("version"),
            "subject_level_balanced_accuracy": meta.get("metrics", {}).get("subject_level_mean_vote", {}).get("balanced_accuracy"),
            "ci95": meta.get("metrics", {}).get("subject_level_balanced_acc_95ci"),
        },
        "required_disclaimer": ("EXPERIMENTAL research pattern comparison only - not a diagnosis, "
                                "not a disease probability, not evidence of having or not having "
                                "Parkinson's disease. The model was validated only on internal "
                                "cross-validation of a small dataset (~0.6 balanced accuracy). "
                                "Microphone and analysis settings differ from the training data. "
                                "Consult a neurologist for any health concern."),
        "validation_stage": 2,
    }
