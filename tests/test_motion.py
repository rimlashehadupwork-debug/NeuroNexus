import numpy as np
import pandas as pd
import pytest
from modules.motion_analysis import analyze_motion


def test_motion_detects_frequency_peak_near_input_frequency():
    fs = 50
    duration = 10
    t = np.arange(0, duration, 1/fs)
    # Acceleration magnitude is non-linear; use a sinusoid on one axis with a constant second axis.
    df = pd.DataFrame({"timestamp": t, "ax": 1.0 + 0.1*np.sin(2*np.pi*5*t), "ay": 0.0, "az": 0.0})
    result = analyze_motion(df)
    assert result["sample_count"] == len(df)
    assert result["sample_rate_hz"] == pytest.approx(fs, rel=0.02)
    assert result["dominant_frequency_hz"] == pytest.approx(5, abs=0.2)


def test_motion_rejects_too_few_samples():
    with pytest.raises(ValueError):
        analyze_motion(pd.DataFrame({"ax": [0, 1], "ay": [0, 1], "az": [0, 1]}))
