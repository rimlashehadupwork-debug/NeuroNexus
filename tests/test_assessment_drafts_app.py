from pathlib import Path

from streamlit.testing.v1 import AppTest

from modules import accounts


def _app_with_draft(tmp_path, monkeypatch):
    monkeypatch.setattr(accounts, "DB_PATH", tmp_path / "drafts.sqlite3")
    accounts.init_db()
    accounts.register("Draft user", "draft@example.test", "long-password")
    user = accounts.authenticate("draft@example.test", "long-password")
    voice_result = {
        "duration_seconds": 5.0,
        "rms": 0.02,
        "silence_ratio": 0.1,
        "pitch_median_hz": 175.0,
        "mfcc_means": [0.0] * 13,
        "quality": {"score": 85, "label": "Good"},
    }
    accounts.save_draft(user["id"], {
        "audio_result": voice_result,
        "reaction_times": [],
    })

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30)
    app.session_state["user"] = user
    app.session_state["workspace_page"] = "Screening assessments"
    app.session_state["consent_granted"] = True
    return app, user, voice_result


def test_screening_page_offers_resume_and_restores_saved_modules(tmp_path, monkeypatch):
    app, _, voice_result = _app_with_draft(tmp_path, monkeypatch)

    app.run()

    assert not app.exception
    assert any(button.label == "Resume Where You Left Off" for button in app.button)
    assert any(button.label == "Start From Beginning" for button in app.button)
    next(button for button in app.button if button.label == "Resume Where You Left Off").click()
    app.run()

    assert not app.exception
    assert app.session_state["audio_result"] == voice_result
    assert app.session_state["voice_wav_bytes"] is None


def test_start_from_beginning_removes_local_draft(tmp_path, monkeypatch):
    app, user, _ = _app_with_draft(tmp_path, monkeypatch)

    app.run()
    next(button for button in app.button if button.label == "Start From Beginning").click()
    app.run()

    assert not app.exception
    assert accounts.load_draft(user["id"]) is None
