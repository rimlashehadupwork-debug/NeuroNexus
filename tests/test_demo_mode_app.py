from pathlib import Path
import time

from streamlit.testing.v1 import AppTest

from modules import accounts


def _click_widget(widgets, label):
    next(widget for widget in widgets if widget.label == label).click().run()


def test_demo_journey_stays_isolated_and_resets_to_initial_state(
    monkeypatch, tmp_path
):
    def reject_database_access(*_args, **_kwargs):
        raise AssertionError("Demo Mode must not access persistent account storage.")

    monkeypatch.setattr(accounts, "DB_PATH", tmp_path / "demo-journey.sqlite3")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(accounts, "init_db", reject_database_access)
    monkeypatch.setattr(accounts, "connect", reject_database_access)
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=90)

    app.run()
    assert not app.exception
    app.button(key="enter_demo_mode").click().run()
    assert not app.exception
    assert app.session_state["user"]["demo_mode"] is True

    _click_widget(app.button, "Start screening")
    assert app.session_state["workspace_page"] == "Screening assessments"
    app.checkbox(key="assessment_consent_checked").check().run()
    assert not app.exception
    next(
        button for button in app.button
        if button.label == "Continue to assessment"
    ).click().run()
    assert not app.exception

    _click_widget(app.button, "Load synthetic demo voice sample")
    assert not app.exception
    _click_widget(app.button, "Analyze synthetic demo sample")
    assert not app.exception
    assert app.session_state["audio_result"] is not None
    assert app.session_state["audio_result"]["sample_rate"] == 16000
    assert app.session_state["audio_result"]["source_channels"] == 1
    assert app.session_state["voice_visualization"]["spectrogram_db"].shape[0] == 257
    assert not app.audio_input
    assert not app.file_uploader
    upload_generation = app.session_state["upload_generation"]
    _click_widget(app.button, "Clear uploaded inputs and retained audio")
    assert not app.exception
    assert app.session_state["audio_result"] is not None
    assert app.session_state["voice_wav_bytes"] is None
    assert app.session_state["upload_generation"] == upload_generation + 1

    _click_widget(app.button, "Load synthetic demo vowel sample")
    assert not app.exception

    _click_widget(app.button, "Start memory assessment")
    app.session_state["memory_check_started"] = time.perf_counter() - 12
    app.session_state["memory_check_presentation_started"] = time.perf_counter() - 11
    app.run()
    assert app.session_state["memory_check_phase"] == "recall"
    remembered_word = app.session_state["memory_check_targets"][0]
    next(
        field for field in app.text_input
        if field.label == "Words you remember"
    ).set_value(remembered_word)
    _click_widget(app.form_submit_button, "Submit answers")
    assert not app.exception
    assert app.session_state["memory_result"]["items_recalled"] == 1

    _click_widget(app.button, "Save report to demo timeline")
    assert not app.exception
    assert len(app.session_state["demo_sessions"]) == 5
    latest_report = app.session_state["demo_sessions"][0]["summary"]
    assert '"demo_mode": true' in latest_report
    assert (
        app.session_state["last_saved_report_session_id"]
        == app.session_state["demo_sessions"][0]["id"]
    )
    assert (
        app.session_state["last_saved_report_created_at"]
        == app.session_state["demo_sessions"][0]["created_at"]
    )

    app.radio(key="workspace_page").set_value("My reports").run()
    assert not app.exception
    assert any(element.label == "Download PDF" for element in app.download_button)
    assert any(
        "demo reports are isolated" in element.value.casefold()
        for element in app.caption
        if isinstance(element.value, str)
    )

    app.radio(key="workspace_page").set_value("Wellness coach").run()
    _click_widget(app.button, "Save wellness plan and today’s activity")
    assert not app.exception
    assert any(element.value == "Clinician visit checklist" for element in app.subheader)
    assert app.session_state["demo_wellness_plan"]["goals"] == []

    app.radio(key="workspace_page").set_value("My dashboard").run()
    _click_widget(app.button, "Reset demo data")
    assert not app.exception
    assert app.session_state["workspace_page"] == "My dashboard"
    assert app.session_state["user"]["demo_mode"] is True
    assert len(app.session_state["demo_sessions"]) == 4
    assert app.session_state["audio_result"] is None
    assert app.session_state["memory_result"] is None
