from io import BytesIO

import numpy as np
import pytest
import soundfile as sf

from modules.audio_analysis import (
    MAX_AUDIO_UPLOAD_BYTES,
    analyze_audio_bytes,
    audio_visualization_data,
    decode_audio_bytes,
    to_standard_wav,
)


def _wav_bytes(sample_rate=22050, seconds=6, channels=1):
    time_axis = np.arange(sample_rate * seconds) / sample_rate
    signal = (0.15 * np.sin(2 * np.pi * 180 * time_axis)).astype("float32")
    if channels > 1:
        signal = np.column_stack((signal, signal * 0.8))
    buffer = BytesIO()
    sf.write(buffer, signal, sample_rate, format="WAV")
    return buffer.getvalue()


def test_audio_analysis_returns_existing_descriptive_features_and_progress():
    stages = []
    out = analyze_audio_bytes(
        _wav_bytes(),
        progress_callback=lambda stage, state, message: stages.append((stage, state)),
    )
    assert out["duration_seconds"] >= 5.9
    assert len(out["mfcc_means"]) == 13
    assert "quality" in out
    assert ("preprocessing", "complete") in stages
    assert ("feature_extraction", "complete") in stages


def test_conversion_reports_original_audio_metadata_and_real_stages():
    stages = []
    wav_bytes, info = to_standard_wav(
        _wav_bytes(channels=2),
        progress_callback=lambda stage, state, message: stages.append((stage, state)),
    )
    result = analyze_audio_bytes(wav_bytes)
    assert info["source_sample_rate"] == 22050
    assert info["source_channels"] == 2
    assert info["target_sample_rate"] == 16000
    assert info["target_channels"] == 1
    assert result["sample_rate"] == 16000
    assert ("validation", "complete") in stages
    assert ("conversion", "complete") in stages


def test_quality_checks_use_original_rate_after_resampling():
    raw = _wav_bytes(sample_rate=8000)
    wav_bytes, info = to_standard_wav(raw)
    result = analyze_audio_bytes(
        wav_bytes,
        source_sample_rate=info["source_sample_rate"],
    )
    assert result["sample_rate"] == 16000
    assert any("Source sample rate is below 16 kHz" in reason for reason in result["quality"]["reasons"])


def test_audio_upload_rejects_files_over_size_limit_before_decoding():
    with pytest.raises(ValueError, match="25 MB upload limit"):
        decode_audio_bytes(b"x" * (MAX_AUDIO_UPLOAD_BYTES + 1))


@pytest.mark.parametrize(
    ("audio_bytes", "message"),
    [
        (b"", "audio input is empty"),
        (b"not an audio container", "Could not decode this audio file"),
    ],
)
def test_audio_upload_rejects_empty_and_unsupported_content(audio_bytes, message):
    with pytest.raises(ValueError, match=message):
        decode_audio_bytes(audio_bytes)


def test_audio_conversion_failure_is_not_replaced_with_fake_audio(monkeypatch):
    def fail_resample(*args, **kwargs):
        raise RuntimeError("simulated resampler failure")

    monkeypatch.setattr("modules.audio_analysis.librosa.resample", fail_resample)

    with pytest.raises(RuntimeError, match="simulated resampler failure"):
        to_standard_wav(_wav_bytes(sample_rate=22050))


def test_audio_upload_rejects_overlong_recordings():
    long_audio = np.zeros(1210, dtype=np.float32)
    buffer = BytesIO()
    sf.write(buffer, long_audio, 10, format="WAV")
    with pytest.raises(ValueError, match="120 second duration limit"):
        decode_audio_bytes(buffer.getvalue())


def test_visualization_data_is_bounded_and_contains_waveform_and_spectrogram():
    visualization = audio_visualization_data(_wav_bytes(sample_rate=16000))
    assert len(visualization["waveform_time_seconds"]) <= 3200
    assert len(visualization["waveform_time_seconds"]) == len(
        visualization["waveform_amplitude"]
    )
    assert visualization["spectrogram_db"].shape[0] == 257
    assert visualization["spectrogram_db"].shape[1] <= 240


def test_silent_audio_spectrogram_remains_finite():
    buffer = BytesIO()
    sf.write(buffer, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
    visualization = audio_visualization_data(buffer.getvalue())
    assert np.isfinite(visualization["spectrogram_db"]).all()
