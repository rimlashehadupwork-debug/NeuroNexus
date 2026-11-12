import numpy as np
import pytest

from modules import praat_features


def test_missing_parselmouth_reports_actionable_error(monkeypatch):
    monkeypatch.setattr(praat_features, "parselmouth", None)

    with pytest.raises(RuntimeError, match="requires praat-parselmouth"):
        praat_features._praat_features(np.zeros(16000, dtype=np.float32), 16000)
