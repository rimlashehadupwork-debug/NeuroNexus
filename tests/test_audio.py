import io
import numpy as np
import soundfile as sf
from modules.audio_analysis import analyze_audio_bytes


def test_audio_analyzer_returns_features():
    sr = 16000
    t = np.arange(sr * 6) / sr
    audio = (0.1 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV")
    result = analyze_audio_bytes(buffer.getvalue())
    assert result["duration_seconds"] == 6
    assert len(result["mfcc_means"]) == 13
    assert result["sample_rate"] == sr
