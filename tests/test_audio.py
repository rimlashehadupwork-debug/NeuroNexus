from io import BytesIO
import numpy as np
import soundfile as sf
from modules.audio_analysis import analyze_audio_bytes

def test_audio_analysis_returns_features():
    sr=22050; t=np.arange(sr*6)/sr
    signal=(0.15*np.sin(2*np.pi*180*t)).astype("float32")
    buffer=BytesIO(); sf.write(buffer, signal, sr, format="WAV")
    out=analyze_audio_bytes(buffer.getvalue())
    assert out["duration_seconds"] >= 5.9
    assert len(out["mfcc_means"]) == 13
    assert "quality" in out
