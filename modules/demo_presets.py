"""Synthetic, non-clinical examples for demonstrating assessment displays."""
from copy import deepcopy

from .confidence import confidence_report


_CLOCK_FEATURES = {
    "stroke_count": 14,
    "path_segment_count": 46,
    "path_endpoint_count": 60,
    "approx_path_length_px": 1065.4,
    "mean_stroke_length_px": 76.1,
    "bounding_box_width_px": 218.0,
    "bounding_box_height_px": 218.0,
    "bounding_box_area_fraction": 0.269,
    "mean_direction_change_degrees": 31.6,
    "drawing_duration_seconds": None,
    "pause_or_air_time_seconds": None,
    "average_stroke_speed_px_per_second": None,
    "pressure": None,
    "jerk": None,
    "timing_limitation": (
        "Synthetic example only. Real canvas exports have no trustworthy pointer "
        "timestamps or pressure data; duration, pauses, speed, pressure, and jerk "
        "are unavailable."
    ),
}


def _audio_result(profile):
    if profile == "clean_signals":
        quality = confidence_report([], 5, 5)
        duration, rms, silence, pitch = 8.2, 0.032, 0.08, 172.0
    else:
        quality = confidence_report(
            ["Synthetic example includes low-level and intermittent audio."], 4, 5
        )
        duration, rms, silence, pitch = 6.8, 0.008, 0.31, 188.0
    return {
        "duration_seconds": duration,
        "sample_rate": 16000,
        "peak_amplitude": 0.21 if profile == "clean_signals" else 0.09,
        "rms": rms,
        "clipping_ratio": 0.0,
        "silence_ratio": silence,
        "mfcc_means": [12.4, -4.1, 2.7, -1.2, 0.8, -0.5, 0.3, -0.2, 0.1, 0.0, -0.1, 0.2, -0.1],
        "mfcc_stds": [18.2, 10.1, 7.4, 5.2, 4.1, 3.6, 3.0, 2.7, 2.3, 2.0, 1.8, 1.6, 1.4],
        "zero_crossing_rate": 0.074 if profile == "clean_signals" else 0.112,
        "spectral_centroid_hz": 1480.0 if profile == "clean_signals" else 1920.0,
        "rms_variability": 0.009 if profile == "clean_signals" else 0.018,
        "pitch_median_hz": pitch,
        "quality": quality,
        "source": "synthetic_demo",
    }


def _motion_result(profile):
    frequencies = [round(index * 0.1, 2) for index in range(1, 101)]
    peak = 2.1 if profile == "clean_signals" else 3.4
    spectrum = [
        round(0.012 / (1 + ((frequency - peak) / 0.35) ** 2), 6)
        for frequency in frequencies
    ]
    return {
        "sample_count": 500,
        "duration_seconds": 10.0,
        "sample_rate_hz": 50.0,
        "timing_jitter_ratio": 0.01 if profile == "clean_signals" else 0.09,
        "dominant_frequency_hz": peak,
        "frequencies_hz": frequencies,
        "spectrum": spectrum,
        "quality": confidence_report([], 4, 4),
        "assumed_sample_rate": False,
        "source": "synthetic_demo",
    }


def _gait_result(profile):
    variable = profile == "variable_signals"
    return {
        "sample_count": 800,
        "duration_seconds": 16.0,
        "sample_rate_hz": 50.0,
        "candidate_step_events": 22 if not variable else 18,
        "cadence_steps_per_min": 104.0 if not variable else 86.0,
        "step_interval_variability_percent": 7.5 if not variable else 18.0,
        "assumed_sample_rate": False,
        "quality": confidence_report([], 4, 4),
        "method_note": (
            "Synthetic demo values; not measured from a walk and not a validated "
            "step detector or clinical interpretation."
        ),
        "source": "synthetic_demo",
    }


def _speech_result(profile):
    features = (
        {
            "n_words": 38,
            "n_unique_words": 28,
            "type_token_ratio": 0.73684,
            "pronoun_ratio": 0.10526,
            "idea_density": 0.52632,
            "mean_word_length": 4.3,
            "n_repeated_bigrams": 0,
        }
        if profile == "clean_signals"
        else {
            "n_words": 31,
            "n_unique_words": 21,
            "type_token_ratio": 0.67742,
            "pronoun_ratio": 0.16129,
            "idea_density": 0.45161,
            "mean_word_length": 4.1,
            "n_repeated_bigrams": 1,
        }
    )
    return {
        "features": features,
        "audio_used": False,
        "model_available": False,
        "message": "No compatible Alzheimer's speech model is installed.",
        "note": (
            "Synthetic demo values only. Speech feature extraction is descriptive; "
            "no individual risk score is calculated."
        ),
        "source": "synthetic_demo",
    }


def load_demo_preset(state, profile):
    """Populate display-only examples; never invokes a trained model."""
    if profile not in ("clean_signals", "variable_signals"):
        raise ValueError(f"Unknown demo profile: {profile}")

    reaction_times = (
        [312.0, 298.0, 325.0, 304.0, 318.0]
        if profile == "clean_signals"
        else [276.0, 438.0, 352.0, 491.0, 319.0]
    )
    state.update({
        "audio_result": _audio_result(profile),
        "motion_result": _motion_result(profile),
        "gait_result": _gait_result(profile),
        "clock_result": {
            "completed": True,
            "task": "Synthetic clock-drawing geometry example",
            "scoring": "No clinical score is produced.",
            "features": deepcopy(_CLOCK_FEATURES),
            "source": "synthetic_demo",
        },
        "reaction_times": reaction_times,
        "reaction_times_demo": True,
        "alz_result": _speech_result(profile),
        "pattern_result": None,
        "voice_wav_bytes": None,
        "demo_profile": (
            "Clean signal demo"
            if profile == "clean_signals"
            else "Variable signal demo"
        ),
        "reaction_started": None,
        "reaction_ready": False,
        "upload_generation": int(state.get("upload_generation", 0)) + 1,
    })
    return state["demo_profile"]


def synthetic_demo_modules(state):
    """List only outputs that still contain values from a demo preset."""
    modules = [
        (key, label)
        for key, label in (
            ("audio_result", "Voice"),
            ("motion_result", "Resting motion"),
            ("gait_result", "Gait"),
            ("clock_result", "Clock drawing"),
            ("alz_result", "Speech features"),
        )
        if (state.get(key) or {}).get("source") == "synthetic_demo"
    ]
    if state.get("reaction_times_demo"):
        modules.append(("reaction_times", "Reaction time"))
    return [label for _, label in modules]


def synthetic_demo_report(profile, modules):
    """Return a disclosure object for reports that still contain demo values."""
    if not modules:
        return None
    return {
        "profile": profile,
        "modules": list(modules),
        "notice": (
            "Synthetic demonstration values only. These values were not collected "
            "from a person and no trained model inference was run on them."
        ),
    }
