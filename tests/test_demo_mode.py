import json

import pytest

from modules import accounts
from modules.demo_mode import (
    enter_demo_mode,
    reset_demo_mode,
    save_demo_report,
    sessions_for_user,
)


def test_demo_history_and_reports_are_session_only_and_resettable(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(accounts, "DB_PATH", tmp_path / "demo.sqlite3")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    accounts.init_db()
    accounts.register("Real participant", "real@example.test", "long-password")
    real_user = accounts.authenticate("real@example.test", "long-password")
    accounts.save_session(real_user["id"], {"voice": {"duration_seconds": 4.0}})
    real_sessions_before = accounts.user_sessions(real_user["id"])
    persistent_reader = accounts.user_sessions
    monkeypatch.setattr(
        accounts,
        "user_sessions",
        lambda *_args, **_kwargs: pytest.fail("Demo Mode attempted persistent session access."),
    )

    state = {"user": None}
    enter_demo_mode(state)
    demo_sessions = sessions_for_user(state["user"], state)

    assert len(demo_sessions) == 4
    assert all(json.loads(row["summary"])["demo_mode"] for row in demo_sessions)
    demo_session_id = save_demo_report(state, {"purpose": "demo report"})
    assert any(row["id"] == demo_session_id for row in state["demo_sessions"])

    reset_demo_mode(state)

    assert state["workspace_page"] == "My dashboard"
    assert state["memory_result"] is None
    assert len(sessions_for_user(state["user"], state)) == 4
    assert persistent_reader(real_user["id"]) == real_sessions_before


def test_demo_entry_refuses_to_replace_an_authenticated_user():
    state = {"user": {"id": 123, "role": "individual"}}

    with pytest.raises(ValueError, match="Sign out"):
        enter_demo_mode(state)


def test_demo_report_saving_requires_demo_mode():
    with pytest.raises(ValueError, match="active Demo Mode"):
        save_demo_report({"user": {"id": 1}}, {"purpose": "not demo"})
