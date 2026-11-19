import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from modules import accounts
from modules.longitudinal import (
    TIMELINE_METRICS,
    is_synthetic_session,
    session_metric_values,
    session_report,
    timeline_rows,
)
from modules.wellness import (
    build_wellness_plan,
    suggested_activities,
    wellness_progress_rows,
)


def test_login_landing_renders_isolated_demo_entry(tmp_path, monkeypatch):
    def reject_database_initialization():
        raise AssertionError("Entering Demo Mode must not initialize persistent storage.")

    monkeypatch.setattr(accounts, "DB_PATH", tmp_path / "login.sqlite3")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(accounts, "init_db", reject_database_initialization)
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=45)

    app.run()

    assert not app.exception
    assert any(
        button.label == "Enter Demo Mode"
        for button in app.button
    )
    next(
        button for button in app.button
        if button.label == "Enter Demo Mode"
    ).click()
    app.run()

    assert not app.exception
    assert app.session_state["workspace_page"] == "My dashboard"
    assert app.session_state["user"]["demo_mode"] is True
    assert app.session_state["user"]["id"] is None
    assert any("demo mode" in element.value.casefold() for element in app.warning)


def test_timeline_extracts_only_present_finite_measurements():
    rows = timeline_rows([
        {
            "created_at": "2026-10-01T10:00:00+00:00",
            "summary": {
                "voice_pattern_features": {"std_pitch": 7.2},
                "gait": {"cadence_steps_per_min": 104},
                "cognitive_tasks": {
                    "reaction_time": {"median_ms": 310},
                    "memory_check": {"items_recalled": 2},
                },
                "clock_drawing_measurements": {
                    "bounding_box_area_fraction": 0.32,
                },
                "demo_history": True,
            },
        },
        {
            "created_at": "2026-10-08T10:00:00+00:00",
            "summary": {"gait": {"cadence_steps_per_min": float("nan")}},
        },
    ])

    assert len(rows) == len(TIMELINE_METRICS)
    assert {row["Measurement"] for row in rows} == set(TIMELINE_METRICS)
    assert all(row["Synthetic demo"] for row in rows)
    assert all(math.isfinite(row["Recorded value"]) for row in rows)


def test_timeline_helpers_preserve_missing_values_and_require_explicit_demo_flags():
    report = {
        "voice_pattern_features": {"std_pitch": 7.2},
        "gait": {"cadence_steps_per_min": float("nan")},
    }

    assert session_metric_values(report) == {
        "Vocal pitch stability (Hz variation)": 7.2,
    }
    assert not is_synthetic_session(report)
    assert is_synthetic_session({"synthetic_demo": {"profile": "sample"}})
    assert session_report({"summary": None}) == {}
    with pytest.raises(ValueError, match="valid JSON"):
        session_report({"summary": "not-json"})


def test_individual_timeline_page_shows_saved_sessions_and_details(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(accounts, "DB_PATH", tmp_path / "timeline.sqlite3")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    accounts.init_db()
    accounts.register("Timeline Participant", "timeline@example.test", "long-password")
    user = accounts.authenticate("timeline@example.test", "long-password")
    now = datetime.now(timezone.utc)
    for days_ago, reaction_ms, recalled in ((7, 320, 2), (1, 305, 3)):
        accounts.save_session(
            user["id"],
            {
                "workflow_mode": "Individual",
                "purpose": "Exploratory signal measurements only.",
                "voice": {
                    "duration_seconds": 18.5,
                    "pitch_median_hz": 145.0,
                    "quality": {"label": "Good"},
                },
                "cognitive_tasks": {
                    "reaction_time": {"trials": 5, "median_ms": reaction_ms},
                    "memory_check": {
                        "completed": True,
                        "items_recalled": recalled,
                        "items_presented": 5,
                    },
                },
            },
            created_at=(now - timedelta(days=days_ago)).isoformat(),
        )

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=60)
    app.session_state["user"] = user
    app.session_state["workspace_page"] = "Brain health timeline"

    app.run()

    assert not app.exception
    assert any(button.label == "View details" for button in app.button)
    assert any(element.label == "Session type" for element in app.selectbox)
    assert any(element.label == "Measurement" for element in app.selectbox)
    assert any(button.label == "Download session JSON" for button in app.download_button)
    assert any(button.label == "Download session PDF" for button in app.download_button)
    assert any(
        "No direction is interpreted as better or worse" in element.value
        for element in app.caption
    )
    detail_buttons = [button for button in app.button if button.label == "View details"]
    detail_buttons[-1].click()
    app.run()
    assert not app.exception
    assert app.session_state["timeline_page_timeline_selected"].endswith("-1")


def test_timeline_page_shows_empty_state_without_creating_history(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(accounts, "DB_PATH", tmp_path / "timeline-empty.sqlite3")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    accounts.init_db()
    accounts.register("New Participant", "new@example.test", "long-password")
    user = accounts.authenticate("new@example.test", "long-password")
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=45)
    app.session_state["user"] = user
    app.session_state["workspace_page"] = "Brain health timeline"

    app.run()

    assert not app.exception
    assert any(
        "Your assessment history will appear here as you complete sessions."
        in element.value
        for element in app.info
    )
    assert any(button.label == "Start an assessment" for button in app.button)
    assert accounts.user_sessions(user["id"]) == []


def test_wellness_plan_uses_optional_self_report_and_domain_coverage():
    plan = build_wellness_plan(
        ["Voice", "Memory check"],
        {
            "sleep_quality": "Often difficult",
            "physical_activity": "Prefer not to say",
            "hydration": "Usually",
            "mental_engagement": "Limited",
        },
    )

    assert len(plan["habit_ideas"]) == 3
    assert any("wind-down" in idea for idea in plan["habit_ideas"])
    assert any("Memory check" in item for item in plan["clinician_checklist"])
    assert any("medicines and supplements" in item for item in plan["clinician_checklist"])


def test_wellness_goals_filter_activities_and_progress_preserves_unlogged_days():
    activities = suggested_activities(["sleep", "social"])
    assert {activity["goal"] for activity in activities} == {"sleep", "social"}
    rows = wellness_progress_rows(
        {
            "2026-10-01": ["sleep_routine"],
            "2026-10-03": [],
        },
        today=datetime(2026, 10, 3).date(),
        days=3,
    )

    assert [row["Activities marked complete"] for row in rows] == [1, None, 0]


def test_wellness_plan_persistence_is_per_user_and_deletable(tmp_path, monkeypatch):
    monkeypatch.setattr(accounts, "DB_PATH", tmp_path / "wellness-storage.sqlite3")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    accounts.init_db()
    accounts.register("First User", "first@example.test", "long-password")
    accounts.register("Second User", "second@example.test", "long-password")
    first = accounts.authenticate("first@example.test", "long-password")
    second = accounts.authenticate("second@example.test", "long-password")
    saved = {
        "goals": ["sleep"],
        "lifestyle": {"sleep_quality": "Mixed"},
        "completed_by_date": {"2026-10-01": ["sleep_routine"]},
        "saved_at": "2026-10-01T10:00:00+00:00",
    }

    assert accounts.save_wellness_plan(first["id"], saved)
    assert accounts.load_wellness_plan(first["id"]) == saved
    assert accounts.load_wellness_plan(second["id"]) is None
    assert accounts.delete_wellness_plan(first["id"])
    assert accounts.load_wellness_plan(first["id"]) is None


def test_wellness_coach_saves_goals_and_daily_activity_for_account(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(accounts, "DB_PATH", tmp_path / "wellness-app.sqlite3")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    accounts.init_db()
    accounts.register("Wellness User", "wellness@example.test", "long-password")
    user = accounts.authenticate("wellness@example.test", "long-password")
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=45)
    app.session_state["user"] = user
    app.session_state["workspace_page"] = "Wellness coach"

    app.run()
    assert not app.exception
    app.multiselect[0].set_value(["sleep", "social"]).run()
    app.selectbox[0].set_value("Often difficult").run()
    next(
        checkbox for checkbox in app.checkbox
        if checkbox.label == "Choose a consistent sleep routine"
    ).check().run()
    next(
        button for button in app.button
        if button.label == "Save wellness plan and today’s activity"
    ).click().run()

    saved = accounts.load_wellness_plan(user["id"])
    assert not app.exception
    assert set(saved["goals"]) == {"sleep", "social"}
    assert saved["lifestyle"]["sleep_quality"] == "Often difficult"
    assert "sleep_routine" in saved["completed_by_date"][datetime.now().date().isoformat()]
    assert any(element.value == "Your wellness dashboard" for element in app.subheader)


def test_individual_demo_dashboard_renders_timeline_and_workspace_cards(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(accounts, "DB_PATH", tmp_path / "dashboard.sqlite3")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    accounts.init_db()
    user = accounts.authenticate(
        "individual@neuronexus.local",
        accounts.DEMO_PASSWORD,
    )
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=45)
    app.session_state["user"] = user
    app.session_state["workspace_page"] = "My dashboard"

    app.run()

    assert not app.exception
    assert any(element.value == "Brain health timeline" for element in app.subheader)
    assert any(element.value == "Latest saved report" for element in app.subheader)
    assert any(element.value == "Privacy controls" for element in app.subheader)
    assert any(button.label == "Open brain health timeline" for button in app.button)
    assert any(button.label == "Open wellness coach" for button in app.button)
    assert any(element.label == "Measurement" for element in app.selectbox)
    assert any(element.label == "Time window" for element in app.selectbox)
    next(button for button in app.button if button.label == "Open wellness coach").click()
    app.run()
    assert not app.exception
    assert app.session_state["workspace_page"] == "Wellness coach"


def test_wellness_coach_has_nonmedical_copy_and_urgent_care_guidance(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(accounts, "DB_PATH", tmp_path / "wellness.sqlite3")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    accounts.init_db()
    user = accounts.authenticate(
        "individual@neuronexus.local",
        accounts.DEMO_PASSWORD,
    )
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=45)
    app.session_state["user"] = user
    app.session_state["workspace_page"] = "Wellness coach"

    app.run()

    assert not app.exception
    assert any("does not diagnose" in element.value for element in app.info)
    assert any(
        "Seek immediate medical care" in element.value
        for element in app.warning
    )
    assert any(
        button.label == "Save wellness plan and today’s activity"
        for button in app.button
    )
    next(
        button for button in app.button
        if button.label == "Save wellness plan and today’s activity"
    ).click()
    app.run()

    assert not app.exception
    assert any(item.value == "Clinician visit checklist" for item in app.subheader)


def test_profile_rejects_invalid_optional_caregiver_email(tmp_path, monkeypatch):
    monkeypatch.setattr(accounts, "DB_PATH", tmp_path / "profile.sqlite3")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    accounts.init_db()
    user = accounts.authenticate(
        "individual@neuronexus.local",
        accounts.DEMO_PASSWORD,
    )
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=45)
    app.session_state["user"] = user
    app.session_state["workspace_page"] = "My profile"

    app.run()
    app.text_input[1].set_value("not-an-email")
    next(
        button for button in app.form_submit_button
        if button.label == "Save profile"
    ).click()
    app.run()

    assert not app.exception
    assert any("valid caregiver email" in element.value for element in app.error)


def test_caregiver_demo_lands_on_synthetic_timeline(tmp_path, monkeypatch):
    monkeypatch.setattr(accounts, "DB_PATH", tmp_path / "caregiver.sqlite3")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    accounts.init_db()
    user = accounts.authenticate(
        "caregiver@neuronexus.local",
        accounts.DEMO_PASSWORD,
    )
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=45)
    app.session_state["user"] = user
    app.session_state["workspace_page"] = "Caregiver overview"

    app.run()

    assert not app.exception
    assert any(element.value == "Brain health timeline" for element in app.subheader)
    assert any(element.label == "Measurement" for element in app.selectbox)
    assert any(element.label == "Time window" for element in app.selectbox)
    assert any(
        "does not represent a linked participant" in element.value
        for element in app.info
    )
