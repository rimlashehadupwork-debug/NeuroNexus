"""Descriptive extraction of comparable measurements from saved reports."""
from __future__ import annotations

import json
import math


TIMELINE_METRICS = {
    "Vocal pitch stability (Hz variation)": (
        "voice_pattern_features",
        "std_pitch",
    ),
    "Movement / gait cadence (steps/min)": (
        "gait",
        "cadence_steps_per_min",
    ),
    "Reaction time (ms)": (
        "cognitive_tasks",
        "reaction_time",
        "median_ms",
    ),
    "Clock drawing compactness (canvas fraction)": (
        "clock_drawing_measurements",
        "bounding_box_area_fraction",
    ),
    "Memory check recall (items)": (
        "cognitive_tasks",
        "memory_check",
        "items_recalled",
    ),
}


def _metric_value(report, label, path):
    if label == "Clock drawing compactness (canvas fraction)":
        value = _nested_value(report, path)
        if value is not None:
            return value
        return _nested_value(
            report,
            ("cognitive_tasks", "clock_drawing", "features", "bounding_box_area_fraction"),
        )
    return _nested_value(report, path)


def session_report(session):
    """Return a saved report as an object, rejecting malformed stored summaries."""
    report = session.get("summary")
    if report is None or report == "":
        return {}
    if isinstance(report, str):
        try:
            report = json.loads(report)
        except json.JSONDecodeError as exc:
            raise ValueError("A saved session summary is not valid JSON.") from exc
    if not isinstance(report, dict):
        raise ValueError("A saved session summary must be an object.")
    return report


def session_metric_values(report):
    """Return only finite, present measurements that are suitable for comparison."""
    return {
        label: value
        for label, path in TIMELINE_METRICS.items()
        if (value := _metric_value(report, label, path)) is not None
    }


def is_synthetic_session(report):
    """Identify explicitly marked demo/sample reports without inferring from values."""
    return bool(
        report.get("demo_history")
        or report.get("demo_mode")
        or report.get("synthetic_demo")
    )


def _nested_value(report, path):
    value = report
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def timeline_rows(sessions):
    """Return one dated descriptive point per available metric and report."""
    rows = []
    for session in sessions:
        report = session_report(session)
        timestamp = session.get("created_at") or report.get("created_utc")
        for label, value in session_metric_values(report).items():
            if timestamp:
                rows.append({
                    "Session date": timestamp,
                    "Measurement": label,
                    "Recorded value": value,
                    "Synthetic demo": is_synthetic_session(report),
                })
    return rows
