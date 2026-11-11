import pytest

from modules.demo_presets import (
    load_demo_preset,
    synthetic_demo_modules,
    synthetic_demo_report,
)
from modules.session import assessment_completion


@pytest.mark.parametrize(
    ("profile", "label"),
    [
        ("clean_signals", "Clean signal demo"),
        ("variable_signals", "Variable signal demo"),
    ],
)
def test_demo_populates_synthetic_measurements_without_model_output(profile, label):
    state = {"upload_generation": 2}

    loaded_label = load_demo_preset(state, profile)

    assert loaded_label == label
    assert all(assessment_completion(state).values())
    assert len(state["audio_result"]["mfcc_means"]) == 13
    assert state["clock_result"]["features"]["pause_or_air_time_seconds"] is None
    assert len(state["reaction_times"]) == 5
    assert state["pattern_result"] is None
    assert state["upload_generation"] == 3
    assert set(synthetic_demo_modules(state)) == {
        "Voice",
        "Resting motion",
        "Gait",
        "Clock drawing",
        "Reaction time",
        "Speech features",
    }
    demo_report = synthetic_demo_report(state["demo_profile"], synthetic_demo_modules(state))
    assert "no trained model inference was run" in demo_report["notice"]


def test_demo_preset_rejects_unknown_profile():
    with pytest.raises(ValueError, match="Unknown demo profile"):
        load_demo_preset({}, "healthy")


def test_demo_disclosure_tracks_only_modules_still_using_demo_values():
    state = {"upload_generation": 0}
    load_demo_preset(state, "clean_signals")
    state["audio_result"] = {"duration_seconds": 4.0}

    assert "Voice" not in synthetic_demo_modules(state)
    demo_report = synthetic_demo_report(state["demo_profile"], synthetic_demo_modules(state))
    assert "Voice" not in demo_report["modules"]
