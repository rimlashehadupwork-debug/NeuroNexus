from io import BytesIO

import numpy as np
import pytest
import soundfile as sf

from modules import praat_features


def test_missing_parselmouth_reports_actionable_error(monkeypatch):
    monkeypatch.setattr(praat_features, "parselmouth", None)

    with pytest.raises(RuntimeError, match="requires praat-parselmouth"):
        praat_features._praat_features(np.zeros(16000, dtype=np.float32), 16000)


def test_praat_features_extracts_modulated_voiced_sample():
    if praat_features.parselmouth is None:
        pytest.skip("praat-parselmouth is not installed")

    sample_rate = 16000
    time_axis = np.arange(sample_rate * 6, dtype=np.float64) / sample_rate
    fundamental = 178 + 2.2 * np.sin(2 * np.pi * 4.3 * time_axis)
    phase = 2 * np.pi * np.cumsum(fundamental) / sample_rate
    sample = (
        0.16 * np.sin(phase)
        + 0.10 * np.sin(2 * phase)
        + 0.06 * np.sin(3 * phase)
        + 0.04 * np.sin(4 * phase)
        + 0.025 * np.sin(6 * phase)
    ).astype(np.float32)
    buffer = BytesIO()
    sf.write(buffer, sample, sample_rate, format="WAV", subtype="PCM_16")

    result = praat_features.run_symptom_pattern_analysis(buffer.getvalue())
    features = result["compatible_features"]

    assert len(features) == 26
    assert features["ac"] > 0.9
    assert features["median_pitch"] == pytest.approx(178, abs=4)
    assert features["num_pulses"] > 900
    assert features["std_period"] > 0
    assert result["recording_quality"]["score"] >= 60
    assert result["experimental_output"]["pattern_label"]
