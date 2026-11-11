"""Non-diagnostic geometry descriptors for a Streamlit clock drawing."""
from __future__ import annotations

import math
from collections.abc import Mapping


def _finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _path_endpoints(path, scale_x, scale_y):
    points = []
    current = None
    for command in path:
        if not isinstance(command, (list, tuple)) or not command:
            continue
        code = str(command[0]).upper()
        endpoint_index = {"M": 1, "L": 1, "Q": 3, "C": 5}.get(code)
        if endpoint_index is None or len(command) < endpoint_index + 2:
            continue
        x, y = command[endpoint_index:endpoint_index + 2]
        if not (_finite_number(x) and _finite_number(y)):
            continue
        point = (float(x) * scale_x, float(y) * scale_y)
        if code != "M" and current is not None:
            points.append((current, point))
        current = point
    return points


def extract_clock_drawing_features(canvas_json, canvas_width=520, canvas_height=340):
    """Return coarse path geometry; canvas data does not contain reliable timing."""
    if not isinstance(canvas_json, Mapping):
        raise ValueError("No drawing data is available. Draw on the canvas first.")
    objects = canvas_json.get("objects")
    if not isinstance(objects, list):
        raise ValueError("Canvas drawing data has an unsupported format.")

    strokes = []
    bounds = []
    total_length = 0.0
    path_segment_count = 0
    endpoint_count = 0
    turn_angles = []

    for drawing in objects:
        if not isinstance(drawing, Mapping):
            continue
        path = drawing.get("path")
        if not isinstance(path, list):
            continue
        scale_x = drawing.get("scaleX", 1)
        scale_y = drawing.get("scaleY", 1)
        if not (_finite_number(scale_x) and _finite_number(scale_y)):
            continue
        scale_x, scale_y = abs(float(scale_x)), abs(float(scale_y))
        stroke_segments = _path_endpoints(path, scale_x, scale_y)
        if not stroke_segments:
            continue
        strokes.append(stroke_segments)
        path_segment_count += len(stroke_segments)
        endpoint_count += len(stroke_segments) + 1
        stroke_vectors = []
        for start, end in stroke_segments:
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            total_length += length
            stroke_vectors.append((dx, dy))
        for previous, current in zip(stroke_vectors, stroke_vectors[1:]):
            previous_length = math.hypot(*previous)
            current_length = math.hypot(*current)
            if previous_length == 0 or current_length == 0:
                continue
            dot = previous[0] * current[0] + previous[1] * current[1]
            cross = previous[0] * current[1] - previous[1] * current[0]
            turn_angles.append(abs(math.degrees(math.atan2(cross, dot))))

        left, top = drawing.get("left"), drawing.get("top")
        width, height = drawing.get("width"), drawing.get("height")
        if all(_finite_number(value) for value in (left, top, width, height)):
            bounds.append((
                float(left), float(top),
                float(left) + abs(float(width) * scale_x),
                float(top) + abs(float(height) * scale_y),
            ))

    if not strokes:
        raise ValueError("No drawn strokes were found. Draw on the canvas first.")

    if bounds:
        min_x = min(box[0] for box in bounds)
        min_y = min(box[1] for box in bounds)
        max_x = max(box[2] for box in bounds)
        max_y = max(box[3] for box in bounds)
    else:
        all_points = [
            point
            for stroke in strokes
            for segment in stroke
            for point in segment
        ]
        min_x = min(point[0] for point in all_points)
        min_y = min(point[1] for point in all_points)
        max_x = max(point[0] for point in all_points)
        max_y = max(point[1] for point in all_points)

    bbox_width = max(0.0, max_x - min_x)
    bbox_height = max(0.0, max_y - min_y)
    canvas_area = float(canvas_width) * float(canvas_height)
    if canvas_area <= 0:
        raise ValueError("Canvas dimensions must be positive.")
    bbox_area_ratio = bbox_width * bbox_height / canvas_area

    return {
        "stroke_count": len(strokes),
        "path_segment_count": path_segment_count,
        "path_endpoint_count": endpoint_count,
        "approx_path_length_px": round(total_length, 2),
        "mean_stroke_length_px": round(total_length / len(strokes), 2),
        "bounding_box_width_px": round(bbox_width, 2),
        "bounding_box_height_px": round(bbox_height, 2),
        "bounding_box_area_fraction": round(bbox_area_ratio, 6),
        "mean_direction_change_degrees": (
            round(sum(turn_angles) / len(turn_angles), 2) if turn_angles else None
        ),
        "drawing_duration_seconds": None,
        "pause_or_air_time_seconds": None,
        "average_stroke_speed_px_per_second": None,
        "pressure": None,
        "jerk": None,
        "timing_limitation": (
            "The canvas export has no trustworthy pointer timestamps or pressure data; "
            "duration, pauses, speed, pressure, and jerk are unavailable."
        ),
    }


def assess_darwin_feature_compatibility(canvas_features, required_features):
    """Check exact feature-name overlap; do not infer missing tablet kinematics."""
    if not isinstance(canvas_features, Mapping):
        raise ValueError("Canvas feature data must be a mapping.")
    if not isinstance(required_features, (list, tuple)) or not required_features:
        return {
            "compatible": False,
            "mapped_feature_count": 0,
            "required_feature_count": 0,
            "unmapped_features": [],
            "message": "No trained DARWIN feature schema is available.",
        }
    available = {
        name for name, value in canvas_features.items()
        if _finite_number(value)
    }
    required = list(dict.fromkeys(str(name) for name in required_features))
    mapped = sorted(available.intersection(required))
    unmapped = [name for name in required if name not in available]
    compatible = len(unmapped) == 0
    return {
        "compatible": compatible,
        "mapped_feature_count": len(mapped),
        "required_feature_count": len(required),
        "unmapped_features": unmapped,
        "message": (
            "Canvas features match the complete model schema."
            if compatible
            else (
                "Live canvas features do not match the DARWIN tablet feature schema; "
                "no kinematic similarity or individual prediction was calculated."
            )
        ),
    }
