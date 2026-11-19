from pathlib import Path
import json
import time

from streamlit.testing.v1 import AppTest

from modules import accounts, pdf_report


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


def test_assessment_consent_requires_checkbox_and_explicit_continue(tmp_path, monkeypatch):
    app, user, _ = _app_with_draft(tmp_path, monkeypatch)
    app.session_state["consent_granted"] = False
    app.session_state["assessment_consent_checked"] = False

    app.run()

    assert not app.exception
    continue_button = next(
        button for button in app.button
        if button.label == "Continue to assessment"
    )
    assert continue_button.disabled
    assert any(
        button.label == "Not now — return to dashboard"
        for button in app.button
    )
    assert not any(button.label == "Resume Where You Left Off" for button in app.button)

    app.checkbox(key="assessment_consent_checked").check().run()
    assert not app.exception
    continue_button = next(
        button for button in app.button
        if button.label == "Continue to assessment"
    )
    assert not continue_button.disabled

    continue_button.click().run()
    assert not app.exception
    assert app.session_state["consent_granted"] is True
    assert app.session_state["user"]["consent_flags"]["assessment"] is True
    assert any(button.label == "Resume Where You Left Off" for button in app.button)
    assert accounts.load_draft(user["id"]) is not None


def test_unconsented_assessment_progress_is_not_autosaved(tmp_path, monkeypatch):
    app, user, voice_result = _app_with_draft(tmp_path, monkeypatch)
    original_draft = accounts.load_draft(user["id"])
    app.session_state["workspace_page"] = "My dashboard"
    app.session_state["consent_granted"] = False
    app.session_state["audio_result"] = {
        **voice_result,
        "duration_seconds": 9.0,
    }

    app.run()

    assert not app.exception
    assert accounts.load_draft(user["id"]) == original_draft


def test_privacy_page_is_available_in_demo_mode_and_discloses_limitations():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=45)

    app.run()
    app.button(key="enter_demo_mode").click().run()
    app.radio(key="workspace_page").set_value("Privacy & Responsible AI").run()

    assert not app.exception
    assert any(element.value == "Privacy & Responsible AI" for element in app.title)
    assert any(element.value == "Information collected" for element in app.subheader)
    assert any(element.value == "Storage and retention" for element in app.subheader)
    assert any(element.value == "Consent and Demo Mode" for element in app.subheader)
    assert any(
        "sample history" in element.value.casefold()
        for element in app.warning
        if isinstance(element.value, str)
    )
    assert any(
        "research-support prototype" in element.value
        for element in app.info
        if isinstance(element.value, str)
    )


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


def test_screening_type_is_selected_and_reflected_in_current_report(tmp_path, monkeypatch):
    app, _, _ = _app_with_draft(tmp_path, monkeypatch)
    app.session_state["pattern_result"] = {
        "compatible_features": {"std_pitch": 1.2},
        "recording_quality": {"score": 90, "reasons": []},
        "experimental_output": {
            "pattern_label": "patterns closer to dataset controls",
            "model_score": 0.437,
            "threshold": 0.5,
        },
        "model_context": {"subject_level_balanced_accuracy": 0.62},
        "required_disclaimer": "Experimental research output only.",
    }

    app.run()
    assert not app.exception
    assert app.selectbox(key="screening_mode").value == "Individual"

    app.selectbox(key="screening_mode").select("Community health worker").run()

    assert not app.exception
    assert app.session_state["screening_mode"] == "Community health worker"
    report = next(json.loads(element.value) for element in app.json)
    assert report["workflow_mode"] == "Community health worker"
    assert (
        report["model_outputs"]["sustained_vowel_research_pattern"]
        ["experimental_output"]["model_score"]
        == 0.437
    )


def test_pdf_generation_failure_is_reported_without_hiding_json_export(
    tmp_path, monkeypatch
):
    app, _, voice_result = _app_with_draft(tmp_path, monkeypatch)
    app.session_state["audio_result"] = voice_result
    app.session_state["draft_resumed"] = True

    def fail_pdf(*args, **kwargs):
        raise RuntimeError("simulated renderer failure")

    monkeypatch.setattr(pdf_report, "make_pdf", fail_pdf)

    app.run()

    assert not app.exception
    assert any(
        "PDF report could not be generated" in element.value
        for element in app.error
    )
    assert any(button.label == "Download JSON report" for button in app.download_button)


def test_start_from_beginning_removes_local_draft(tmp_path, monkeypatch):
    app, user, _ = _app_with_draft(tmp_path, monkeypatch)

    app.run()
    next(button for button in app.button if button.label == "Start From Beginning").click()
    app.run()

    assert not app.exception
    assert accounts.load_draft(user["id"]) is None


def test_sample_clock_drawing_populates_capture_card(tmp_path, monkeypatch):
    app, _, _ = _app_with_draft(tmp_path, monkeypatch)
    app.session_state["draft_resumed"] = True

    app.run()
    assert not app.exception

    next(
        button for button in app.button
        if button.label == "Load Sample Clock Drawing"
    ).click()
    app.run()
    assert not app.exception

    next(button for button in app.button if button.label == "Analyze drawing").click()
    app.run()

    assert not app.exception
    result = app.session_state["clock_result"]
    assert result["features"]["stroke_count"] > 0
    assert result["features"]["approx_path_length_px"] > 0
    assert result["features"]["bounding_box_area_fraction"] > 0
    assert result["features"]["drawing_duration_seconds"] is None
    assert any(
        item.value == "Geometric & Kinematic Capture Card"
        for item in app.subheader
    )


def test_memory_check_starts_with_five_words_in_consent_gated_assessment(
    tmp_path, monkeypatch
):
    app, _, _ = _app_with_draft(tmp_path, monkeypatch)
    app.session_state["draft_resumed"] = True

    app.run()
    assert not app.exception

    next(
        button for button in app.button
        if button.label == "Start memory assessment"
    ).click()
    app.run()

    assert not app.exception
    assert app.session_state["memory_check_phase"] == "presentation"
    assert len(app.session_state["memory_check_targets"]) == 5


def test_memory_check_flow_auto_hides_words_and_renders_complete_results(
    tmp_path, monkeypatch
):
    app, _, _ = _app_with_draft(tmp_path, monkeypatch)
    app.session_state["draft_resumed"] = True
    app.session_state["memory_check_phase"] = "presentation"
    app.session_state["memory_check_targets"] = [
        "APPLE", "RIVER", "CHAIR", "BLUE", "BOOK",
    ]
    app.session_state["memory_check_started"] = time.perf_counter() - 15
    app.session_state["memory_check_presentation_started"] = time.perf_counter() - 6
    app.session_state["memory_check_presentation_seconds"] = 5
    app.session_state["memory_check_show_timer"] = False

    app.run()
    assert not app.exception

    assert app.session_state["memory_check_phase"] == "recall"

    next(
        field for field in app.text_input
        if field.label == "Words you remember"
    ).set_value("apple, chair!")
    next(
        button for button in app.form_submit_button
        if button.label == "Submit answers"
    ).click()
    app.run()

    assert not app.exception
    assert app.session_state["memory_result"]["items_recalled"] == 2
    assert app.session_state["memory_result"]["items_presented"] == 5
    assert app.session_state["memory_result"]["recall_percentage"] == 40
    assert app.session_state["memory_result"]["words_recalled"] == ["APPLE", "CHAIR"]
    assert app.session_state["memory_result"]["words_not_recalled"] == [
        "RIVER", "BLUE", "BOOK",
    ]
    assert any(metric.label == "Correct words recalled" for metric in app.metric)


def test_memory_check_exit_preserves_result_and_restart_clears_it(tmp_path, monkeypatch):
    app, _, _ = _app_with_draft(tmp_path, monkeypatch)
    app.session_state["draft_resumed"] = True
    app.session_state["memory_check_phase"] = "complete"
    app.session_state["memory_result"] = {
        "completed": True,
        "completion_status": "completed",
        "items_presented": 3,
        "items_recalled": 1,
        "recall_latency_seconds": 2.0,
    }

    app.run()
    assert not app.exception
    next(
        button for button in app.button
        if button.label == "Exit memory check"
    ).click()
    app.run()
    assert not app.exception
    assert app.session_state["memory_check_phase"] == "ready"
    assert app.session_state["memory_result"]["items_recalled"] == 1

    app.session_state["memory_check_phase"] = "complete"
    app.run()
    next(
        button for button in app.button
        if button.label == "Restart memory assessment"
    ).click()
    app.run()
    assert not app.exception
    assert app.session_state["memory_check_phase"] == "ready"
    assert app.session_state["memory_result"] is None
