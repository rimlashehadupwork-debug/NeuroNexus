"""Safe report serialization: only compact derived features, never raw samples."""
import json
from datetime import datetime, timezone

def build_report(audio_result=None, motion_result=None, gait_result=None, mode="Individual"):
    def compact_audio(r):
        if not r: return None
        keys = ("duration_seconds", "sample_rate", "rms", "clipping_ratio", "silence_ratio",
                "pitch_median_hz", "zero_crossing_rate", "spectral_centroid_hz", "rms_variability", "quality")
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
    voice = report.get("voice") or {}
    motion = report.get("resting_motion") or {}
    gait = report.get("gait") or {}
    reaction = cognitive.get("reaction_time") or {}

    def format_value(value, unit=""):
        if value is None or value == "":
            return "Not completed"
        if isinstance(value, float):
            value = f"{value:.4g}"
        return f"{value} {unit}".strip()

    measurements = [
        ("Voice", "Recording duration", voice.get("duration_seconds"), "s"),
        ("Voice", "Median pitch", voice.get("pitch_median_hz"), "Hz"),
        ("Voice", "Silence ratio", voice.get("silence_ratio"), ""),
        ("Speech", "Transcript word count", speech.get("n_words"), "words"),
        ("Speech", "Type-token ratio", speech.get("type_token_ratio"), ""),
        ("Movement", "Dominant motion frequency", motion.get("dominant_frequency_hz"), "Hz"),
        ("Movement", "Estimated cadence", gait.get("cadence_steps_per_min"), "steps/min"),
        ("Clock drawing", "Stroke count", clock.get("stroke_count"), "strokes"),
        ("Clock drawing", "Approximate path length", clock.get("approx_path_length_px"), "px"),
        ("Clock drawing", "Canvas area used", clock.get("bounding_box_area_fraction"), ""),
        ("Reaction time", "Median response", reaction.get("median_ms"), "ms"),
        ("Reaction time", "Response variability", reaction.get("variability_ms"), "ms"),
    ]
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
