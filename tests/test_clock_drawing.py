import pytest

from modules.clock_drawing import (
    assess_darwin_feature_compatibility,
    extract_clock_drawing_features,
)


def test_extracts_only_geometry_from_canvas_strokes():
    drawing = {
        "objects": [
            {
                "type": "path",
                "path": [["M", 0, 0], ["L", 30, 0], ["L", 30, 40]],
                "left": 10,
                "top": 20,
                "width": 30,
                "height": 40,
            },
            {
                "type": "path",
                "path": [["M", 0, 0], ["L", 0, 10]],
                "left": 60,
                "top": 70,
                "width": 0,
                "height": 10,
            },
        ]
    }

    features = extract_clock_drawing_features(drawing, canvas_width=100, canvas_height=100)

    assert features["stroke_count"] == 2
    assert features["path_segment_count"] == 3
    assert features["approx_path_length_px"] == 80
    assert features["bounding_box_area_fraction"] == 0.3
    assert features["mean_direction_change_degrees"] == 90
    assert features["drawing_duration_seconds"] is None
    assert features["pressure"] is None
    assert features["jerk"] is None


@pytest.mark.parametrize("drawing", [None, {}, {"objects": []}])
def test_rejects_missing_or_empty_canvas_strokes(drawing):
    with pytest.raises(ValueError):
        extract_clock_drawing_features(drawing)


def test_canvas_geometry_does_not_claim_compatibility_with_darwin_tablet_schema():
    features = extract_clock_drawing_features({
        "objects": [{
            "type": "path",
            "path": [["M", 0, 0], ["L", 30, 40]],
            "left": 0,
            "top": 0,
            "width": 30,
            "height": 40,
        }]
    })

    compatibility = assess_darwin_feature_compatibility(
        features,
        ["air_time1", "mean_speed_in_air1", "pressure_mean1"],
    )

    assert compatibility["compatible"] is False
    assert compatibility["mapped_feature_count"] == 0
    assert compatibility["required_feature_count"] == 3
    assert len(compatibility["unmapped_features"]) == 3
