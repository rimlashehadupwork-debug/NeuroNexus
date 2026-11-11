"""Tests for the Alzheimer's feature extractor and audio conversion helper."""
import pytest

from modules.alzheimers_model import (
    extract_speech_features,
    get_alzheimers_speech_status,
    run_alzheimers_feature_report,
)


def test_extract_lexical_features_from_transcript():
    feats = extract_speech_features(
        "I remember the park and the trees and the children playing there "
        "with their friends every single Sunday afternoon last summer.")
    assert feats["n_words"] >= 10
    assert 0 < feats["type_token_ratio"] <= 1
    assert 0 <= feats["pronoun_ratio"] < 1
    assert 0 < feats["idea_density"] <= 1


def test_empty_transcript_rejected():
    with pytest.raises(ValueError):
        extract_speech_features("")


def test_speech_report_keeps_kinematic_and_speech_model_status_separate():
    rep = run_alzheimers_feature_report(
        "We walked along the river and watched the boats going by very slowly "
        "on a bright and quiet morning together.")
    assert rep["model_available"] is False
    assert "no individual risk result" in rep["message"].lower()
    assert get_alzheimers_speech_status()["available"] is False


def test_to_standard_wav_decodes_synthesized_wav():
    import io
    import numpy as np
    import soundfile as sf
    sr, secs = 22050, 2
    audio = (0.2 * np.sin(2 * np.pi * 220 * np.linspace(0, secs, sr * secs))).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    from modules.audio_analysis import to_standard_wav, analyze_audio_bytes
    wav_bytes, info = to_standard_wav(buf.getvalue())
    assert info["target_sample_rate"] == 16000
    res = analyze_audio_bytes(wav_bytes)
    assert res["sample_rate"] == 16000
    assert res["duration_seconds"] > 1
