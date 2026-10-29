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
        "product": "NeuroNexus", "version": "1.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_mode": mode,
        "purpose": "Exploratory signal measurements only; not a diagnosis or disease-risk estimate.",
        "voice": compact_audio(audio_result),
        "resting_motion": compact_motion(motion_result),
        "gait": compact_gait(gait_result),
    }
def report_json(**kwargs):
    return json.dumps(build_report(**kwargs), indent=2, allow_nan=False)
