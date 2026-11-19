"""Safe report serialization: only compact derived features, never raw samples."""
import json
import math
import re
from datetime import datetime, timezone


_FEATURE_LABELS = {
    "air_time": "Pen air / pause time",
    "disp_index": "Drawing displacement index",
    "gmrt_in_air": "Pen-air movement timing",
    "gmrt_on_paper": "Pen-on-paper movement timing",
    "max_x_extension": "Maximum horizontal reach",
    "max_y_extension": "Maximum vertical reach",
    "mean_acc_in_air": "Average pen-air acceleration",
    "mean_acc_on_paper": "Average pen-on-paper acceleration",
    "mean_gmrt": "Average movement timing",
    "mean_jerk_in_air": "Pen-air movement smoothness",
    "mean_jerk_on_paper": "Pen-on-paper movement smoothness",
    "mean_speed_in_air": "Pen-air drawing speed",
    "mean_speed_on_paper": "Pen-on-paper drawing speed",
    "num_of_pendown": "Pen-down stroke count",
    "paper_time": "Pen-on-paper time",
    "pressure_mean": "Average pen pressure",
    "pressure_var": "Pen pressure variability",
    "total_time": "Total drawing time",
}


def humanize_feature_code(feature):
    """Translate a dataset feature code into a readable label."""
    code = str(feature)
    match = re.fullmatch(r"(.+?)(\d+)", code)
    if match:
        name, task_number = match.groups()
        label = _FEATURE_LABELS.get(name, name.replace("_", " ").capitalize())
        return f"{label} (task {task_number})"
    return _FEATURE_LABELS.get(code, code.replace("_", " ").capitalize())


def completed_module_explanations(report):
    """Return plain-language descriptions for modules recorded in a report."""
    report = report or {}
    cognitive = report.get("cognitive_tasks") or {}
    clock = cognitive.get("clock_drawing") or {}
    reaction = cognitive.get("reaction_time") or {}
    return {
        label: description
        for label, completed, description in (
            (
                "Voice",
                bool(report.get("voice") or report.get("voice_pattern_features")),
                "Recording duration, sound level, near-silence ratio, spectral features, and estimated vocal pitch.",
            ),
            (
                "Resting motion",
                bool(report.get("resting_motion")),
                "Frequency content and timing in the uploaded resting-sensor recording.",
            ),
            (
                "Gait",
                bool(report.get("gait")),
                "Candidate step events, estimated cadence, and timing variability from uploaded movement data.",
            ),
            (
                "Clock drawing",
                bool(
                    report.get("clock_drawing_measurements")
                    or clock.get("completed")
                    or clock.get("features")
                ),
                "Stroke count, path length, and the portion of the canvas used in your drawing.",
            ),
            (
                "Reaction time",
                bool(reaction.get("trials", 0)),
                "Median response time and variation across button-tap trials; device and browser timing affect results.",
            ),
            (
                "Memory check",
                bool((cognitive.get("memory_check") or {}).get("completed")),
                "Correct words recalled, recall percentage, and completion time after timed word presentation; descriptive only, not a diagnostic test.",
            ),
            (
                "Speech features",
                bool(report.get("alzheimers_speech_features")),
                "Transcript word counts and language patterns; optional audio timing features when matching audio was provided.",
            ),
        )
        if completed
    }


def build_report(audio_result=None, motion_result=None, gait_result=None, mode="Individual"):
    def compact_audio(r):
        if not r: return None
        keys = ("duration_seconds", "sample_rate", "peak_amplitude", "rms",
                "clipping_ratio", "silence_ratio",
                "pitch_median_hz", "zero_crossing_rate", "spectral_centroid_hz",
                "rms_variability", "source_sample_rate", "source_channels",
                "source_duration_seconds", "conversion_method", "quality")
        return {k: r.get(k) for k in keys}
    def compact_motion(r):
        if not r: return None
        keys = ("sample_count", "duration_seconds", "sample_rate_hz", "dominant_frequency_hz",
                "timing_jitter_ratio", "assumed_sample_rate", "quality")
        return {k: r.get(k) for k in keys}
    def compact_gait(r):
        if not r: return None
        keys = ("sample_count", "duration_seconds", "sample_rate_hz", "candidate_step_events",
                "cadence_steps_per_min", "step_interval_variability_percent", "assumed_sample_rate",
                "quality", "method_note")
        return {k: r.get(k) for k in keys}
    return {
        "product": "NeuroNexus", "version": "2.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_mode": mode,
        "purpose": "Exploratory signal measurements only; not a diagnosis or disease-risk estimate.",
        "voice": compact_audio(audio_result),
        "resting_motion": compact_motion(motion_result),
        "gait": compact_gait(gait_result),
    }


def key_measurement_rows(report):
    """Describe recorded measurements without clinical reference ranges or flags."""
    report = report or {}
    cognitive = report.get("cognitive_tasks") or {}
    clock = report.get("clock_drawing_measurements") or {}
    if not clock:
        clock = (cognitive.get("clock_drawing") or {}).get("features") or {}
    speech = report.get("alzheimers_speech_features") or {}
    voice_pattern = report.get("voice_pattern_features") or {}
    voice = report.get("voice") or {}
    motion = report.get("resting_motion") or {}
    gait = report.get("gait") or {}
    reaction = cognitive.get("reaction_time") or {}
    memory = cognitive.get("memory_check") or {}
    recalled_words = memory.get("words_recalled")
    not_recalled_words = memory.get("words_not_recalled")

    def format_value(value, unit=""):
        if value is None or value == "":
            return "Not completed"
        if isinstance(value, float) and not math.isfinite(value):
            return "Not completed"
        if isinstance(value, float):
            value = f"{value:.4g}"
        return f"{value} {unit}".strip()

    measurements = [
        ("Voice", "Recording duration", voice.get("duration_seconds"), "s"),
        ("Voice", "Source sample rate", voice.get("source_sample_rate"), "Hz"),
        ("Voice", "Source channels", voice.get("source_channels"), "channels"),
        ("Voice", "Median pitch", voice.get("pitch_median_hz"), "Hz"),
        ("Voice", "Silence ratio", voice.get("silence_ratio"), ""),
        ("Voice", "RMS signal level", voice.get("rms"), ""),
        ("Voice", "Peak amplitude", voice.get("peak_amplitude"), ""),
        ("Voice", "Clipping ratio", voice.get("clipping_ratio"), ""),
        ("Voice", "Zero-crossing rate", voice.get("zero_crossing_rate"), ""),
        ("Voice", "Spectral centroid", voice.get("spectral_centroid_hz"), "Hz"),
        ("Voice", "RMS frame variability", voice.get("rms_variability"), ""),
        ("Speech", "Transcript word count", speech.get("n_words"), "words"),
        ("Speech", "Type-token ratio", speech.get("type_token_ratio"), ""),
        ("Movement", "Dominant motion frequency", motion.get("dominant_frequency_hz"), "Hz"),
        ("Movement", "Estimated cadence", gait.get("cadence_steps_per_min"), "steps/min"),
        ("Clock drawing", "Stroke count", clock.get("stroke_count"), "strokes"),
        ("Clock drawing", "Approximate path length", clock.get("approx_path_length_px"), "px"),
        ("Clock drawing", "Canvas area used", clock.get("bounding_box_area_fraction"), ""),
        ("Reaction time", "Median response", reaction.get("median_ms"), "ms"),
        ("Reaction time", "Response variability", reaction.get("variability_ms"), "ms"),
        ("Memory check", "Items recalled", memory.get("items_recalled"), "items"),
        ("Memory check", "Items presented", memory.get("items_presented"), "words"),
        ("Memory check", "Recall percentage", memory.get("recall_percentage"), "%"),
        (
            "Memory check",
            "Words recalled",
            ", ".join(recalled_words) if isinstance(recalled_words, list) else None,
            "",
        ),
        (
            "Memory check",
            "Words not recalled",
            ", ".join(not_recalled_words)
            if isinstance(not_recalled_words, list)
            else None,
            "",
        ),
        ("Memory check", "Recall latency", memory.get("recall_latency_seconds"), "s"),
        ("Memory check", "Completion time", memory.get("completion_time_seconds"), "s"),
        ("Memory check", "Completion status", memory.get("completion_status"), ""),
    ]
    if voice_pattern.get("std_pitch") is not None:
        measurements.append(
            (
                "Voice",
                "Vocal pitch stability (variation)",
                voice_pattern["std_pitch"],
                "Hz",
            )
        )
    reaction_completed = bool(reaction.get("trials", 0))
    rows = []
    for domain, label, value, unit in measurements:
        if domain == "Reaction time" and not reaction_completed:
            value = None
        rows.append({
            "Domain": domain,
            "Measurement": label,
            "Recorded value": format_value(value, unit),
            "Reference range": "Not established",
            "Interpretation": "Descriptive only",
        })
    return rows


def report_json(**kwargs):
    return json.dumps(build_report(**kwargs), indent=2, allow_nan=False)
