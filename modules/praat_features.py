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
PRAAT_FEATURE_EXPLANATIONS = {
    "jitter_local": (
        "Average local cycle-to-cycle period variation.",
        "A dataset-compatible descriptor of periodic timing in a sustained vowel.",
    ),
    "jitter_local_abs": (
        "Absolute local cycle-to-cycle period variation in seconds.",
        "A dataset-compatible descriptor of periodic timing in a sustained vowel.",
    ),
    "jitter_rap": (
        "Relative average perturbation over three adjacent periods.",
        "A dataset-compatible descriptor of periodic timing in a sustained vowel.",
    ),
    "jitter_ppq5": (
        "Five-period pitch-period quotient variation.",
        "A dataset-compatible descriptor of periodic timing in a sustained vowel.",
    ),
    "jitter_ddp": (
        "Derived difference-of-differences period variation.",
        "A dataset-compatible descriptor of periodic timing in a sustained vowel.",
    ),
    "shimmer_local": (
        "Local cycle-to-cycle amplitude variation.",
        "A dataset-compatible descriptor of amplitude variation in a sustained vowel.",
    ),
    "shimmer_local_db": (
        "Local cycle-to-cycle amplitude variation in decibels.",
        "A dataset-compatible descriptor of amplitude variation in a sustained vowel.",
    ),
    "shimmer_apq3": (
        "Amplitude perturbation quotient averaged over three periods.",
        "A dataset-compatible descriptor of amplitude variation in a sustained vowel.",
    ),
    "shimmer_apq5": (
        "Amplitude perturbation quotient averaged over five periods.",
        "A dataset-compatible descriptor of amplitude variation in a sustained vowel.",
    ),
    "shimmer_apq11": (
        "Amplitude perturbation quotient averaged over eleven periods.",
        "A dataset-compatible descriptor of amplitude variation in a sustained vowel.",
    ),
    "shimmer_dda": (
        "Derived difference-of-differences amplitude variation.",
        "A dataset-compatible descriptor of amplitude variation in a sustained vowel.",
    ),
    "ac": (
        "Mean strength reported by the Praat pitch tracker for voiced frames.",
        "Provides context about periodicity in the extracted vowel signal.",
    ),
    "nth": (
        "Noise-to-harmonics ratio derived from the harmonicity measurement.",
        "Describes the signal's estimated periodic and noise components.",
    ),
    "htn": (
        "Mean harmonicity estimate in decibels.",
        "Describes the signal's estimated periodic and noise components.",
    ),
    "median_pitch": (
        "Median estimated fundamental frequency of voiced frames.",
        "Summarizes pitch estimates for the sustained-vowel task.",
    ),
    "mean_pitch": (
        "Mean estimated fundamental frequency of voiced frames.",
        "Summarizes pitch estimates for the sustained-vowel task.",
    ),
    "std_pitch": (
        "Standard deviation of estimated fundamental frequency.",
        "Describes variation in pitch estimates within this recording.",
    ),
    "min_pitch": (
        "Minimum estimated fundamental frequency of voiced frames.",
        "Summarizes pitch estimates for the sustained-vowel task.",
    ),
    "max_pitch": (
        "Maximum estimated fundamental frequency of voiced frames.",
        "Summarizes pitch estimates for the sustained-vowel task.",
    ),
    "num_pulses": (
        "Number of pulse events identified by the Praat point-process analysis.",
        "Describes the pulse sequence available to the feature extractor.",
    ),
    "num_periods": (
        "Number of periods identified between pulse events.",
        "Describes the pulse sequence available to the feature extractor.",
    ),
    "mean_period": (
        "Mean duration of identified periods in seconds.",
        "Describes timing in the sustained-vowel pulse sequence.",
    ),
    "std_period": (
        "Standard deviation of identified period durations in seconds.",
        "Describes timing variability in the sustained-vowel pulse sequence.",
    ),
    "frac_unvoiced": (
        "Fraction of pitch frames classified as unvoiced.",
        "Describes how much of the task signal lacked a voiced pitch estimate.",
    ),
    "num_voice_breaks": (
        "Count of interior unvoiced segments between voiced regions.",
        "Describes interruptions in the extracted pitch track.",
    ),
    "degree_voice_breaks": (
        "Fraction of pitch frames in interior unvoiced segments.",
        "Describes interruptions in the extracted pitch track.",
    ),
}


def _emit(progress_callback, stage, state, message):
    if progress_callback is not None:
        progress_callback(stage, state, message)


def to_wav_mono16k(
    audio_bytes: bytes,
    progress_callback=None,
) -> tuple[np.ndarray, int]:
    """Decode any supported audio format to mono 16 kHz float waveform."""
    _emit(progress_callback, "conversion", "running", "Preparing the sustained-vowel signal.")
    audio, sr = sf.read(BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    _emit(
        progress_callback,
        "conversion",
        "complete",
        f"Prepared mono audio at {TARGET_SR} Hz for the existing Praat feature settings.",
    )
    _emit(progress_callback, "preprocessing", "running", "Applying the existing peak normalization.")
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = (audio / peak * 0.9).astype(np.float32)  # gentle normalization
    _emit(
        progress_callback,
        "preprocessing",
        "complete",
        "Applied the same peak normalization used by this research pathway.",
    )
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
    pitch_values = pitch.selected_array
    f0 = pitch_values["frequency"]
    f0v = f0[f0 > 0]
    if f0v.size < sr * MIN_VOICED_SECONDS / 100.0:  # too few voiced frames
        raise ValueError("Too little voiced audio. Sustain a steady vowel such as 'ah' for at least 5 seconds.")
    ac = float(np.mean(pitch_values["strength"][f0 > 0]))
    hnr = call(snd, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)
    hnr_db = call(hnr, "Get mean", 0.0, d)
    rap = call(point, "Get jitter (rap)", 0.0, d, 0.0001, 0.02, 1.3)
    apq3 = call([snd, point], "Get shimmer (apq3)", 0.0, d, 0.0001, 0.02, 1.3, 1.6)
    num_periods = int(call(point, "Get number of periods", 0.0, d, 0.0001, 0.02, 1.3))
    pulse_times = np.asarray([
        call(point, "Get time from index", index)
        for index in range(1, num_periods + 2)
    ])
    period_durations = np.diff(pulse_times)
    unvoiced = f0 <= 0
    unvoiced_edges = np.diff(np.concatenate(([False], unvoiced, [False])).astype(int))
    unvoiced_starts = np.flatnonzero(unvoiced_edges == 1)
    unvoiced_ends = np.flatnonzero(unvoiced_edges == -1)
    interior_breaks = (unvoiced_starts > 0) & (unvoiced_ends < len(unvoiced))
    break_frames = int(np.sum(
        unvoiced_ends[interior_breaks] - unvoiced_starts[interior_breaks]
    ))
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
        "num_pulses": num_periods + 1,
        "num_periods": num_periods,
        "mean_period": call(point, "Get mean period", 0.0, d, 0.0001, 0.02, 1.3),
        "std_period": float(np.std(period_durations)) if period_durations.size else 0.0,
        "frac_unvoiced": float(np.mean(f0 <= 0)),
        "num_voice_breaks": int(np.count_nonzero(interior_breaks)),
        "degree_voice_breaks": float(break_frames / len(unvoiced)),
    }


def extract_voice_features(
    audio_bytes: bytes,
    progress_callback=None,
) -> tuple[np.ndarray, dict]:
    """WAV conversion + 26-feature extraction. Returns (wav, features)."""
    wav, sr = to_wav_mono16k(audio_bytes, progress_callback)
    _emit(
        progress_callback,
        "feature_extraction",
        "running",
        "Extracting the existing Praat-compatible sustained-vowel features.",
    )
    features = _praat_features(wav, sr)
    _emit(
        progress_callback,
        "feature_extraction",
        "complete",
        f"Extracted {len(features)} task-specific features.",
    )
    return wav, features


def run_symptom_pattern_analysis(
    audio_bytes: bytes,
    models_dir: str | Path = "models",
    progress_callback=None,
) -> dict:
    """Full gated pipeline. Returns a dict safe to render in the UI.

    Gates: decodable WAV, >=5 s, passes the app's quality checks, enough voiced
    audio. The model output is labeled experimental and carries its metrics
    and limitations so the UI never presents it as a diagnosis.
    """
    from .audio_analysis import analyze_audio_bytes  # existing quality checks
    from .model_registry import load_parkinson_model

    _emit(
        progress_callback,
        "validation",
        "running",
        "Checking the sustained-vowel sample and technical recording quality.",
    )
    quality = analyze_audio_bytes(audio_bytes)  # raises if undecodable/too short
    _emit(
        progress_callback,
        "validation",
        "complete",
        "Audio decoded and existing technical quality checks completed.",
    )
    wav, feats = extract_voice_features(audio_bytes, progress_callback)

    _emit(
        progress_callback,
        "model_inference",
        "running",
        "Loading the existing research classifier and computing its output.",
    )
    pipeline, feature_columns, threshold = load_parkinson_model(models_dir)
    row = pd.DataFrame([feats], columns=None)[feature_columns]
    prob = float(pipeline.predict_proba(row)[0, 1])
    pred = int(prob >= threshold)
    _emit(
        progress_callback,
        "model_inference",
        "complete",
        "The existing classifier produced a research pattern comparison.",
    )

    _emit(
        progress_callback,
        "results_preparation",
        "running",
        "Assembling descriptive features, classifier context, and limitations.",
    )
    meta_path = Path(models_dir) / "parkinson_voice_research_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    result = {
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
    _emit(
        progress_callback,
        "results_preparation",
        "complete",
        "Research output and its limitations are ready.",
    )
    return result
