import numpy as np
import pandas as pd
import pytest
from modules.motion_analysis import (
    MAX_SENSOR_SAMPLES,
    MAX_SENSOR_UPLOAD_BYTES,
    analyze_gait,
    analyze_motion,
    read_sensor_csv,
)


def test_sensor_csv_is_bounded_and_validated_before_analysis():
    valid_csv = b"timestamp,ax,ay,az\n0,0,0,1\n1,0,0,1\n"
    assert len(read_sensor_csv(valid_csv)) == 2

    with pytest.raises(ValueError, match="5 MB upload limit"):
        read_sensor_csv(b"x" * (MAX_SENSOR_UPLOAD_BYTES + 1))

    large_csv = (
        b"timestamp,ax,ay,az\n"
        + b"0,0,0,1\n" * (MAX_SENSOR_SAMPLES + 1)
    )
    with pytest.raises(ValueError, match="100,000-sample limit"):
        read_sensor_csv(large_csv)

    too_many_columns = (
        ",".join(f"column_{index}" for index in range(65))
        + "\n"
        + ",".join("0" for _ in range(65))
    ).encode()
    with pytest.raises(ValueError, match="too many columns"):
        read_sensor_csv(too_many_columns)

def test_motion_frequency_peak():
    fs=50; t=np.arange(0,10,1/fs)
    df=pd.DataFrame({"timestamp":t,"ax":1+0.1*np.sin(2*np.pi*5*t),"ay":np.zeros_like(t),"az":np.zeros_like(t)})
    out=analyze_motion(df)
    assert abs(out["dominant_frequency_hz"]-5) < 0.5
    assert out["sample_count"] == len(t)

def test_motion_rejects_too_few_samples():
    with pytest.raises(ValueError):
        analyze_motion(pd.DataFrame({"ax":[0]*10,"ay":[0]*10,"az":[0]*10}))

def test_gait_returns_candidate_events():
    fs=50; t=np.arange(0,15,1/fs)
    pulse=np.sin(2*np.pi*1.8*t)
    df=pd.DataFrame({"timestamp":t,"ax":1+0.08*pulse,"ay":0.03*np.sin(2*np.pi*0.9*t),"az":0.02*np.sin(2*np.pi*0.4*t)})
    out=analyze_gait(df)
    assert out["sample_count"] == len(t)
    assert out["candidate_step_events"] >= 1
