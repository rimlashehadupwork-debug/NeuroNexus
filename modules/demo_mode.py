"""Isolated, session-only synthetic data for the hackathon demo experience."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from uuid import uuid4


DEMO_MODE_NOTICE = (
    "Demo Mode uses an isolated synthetic participant and sample history. "
    "It is not a patient record; demo activity is kept only in this browser session."
)
_DEMO_MEMORY_WORDS = ("APPLE", "RIVER", "CHAIR", "BLUE", "BOOK")


def build_synthetic_history(now=None):
    """Return dated synthetic session summaries without touching persistent storage."""
    measurements = (
        (35, 7.8, 104.0, 322.0, 0.31, 2),
        (28, 6.9, 108.0, 310.0, 0.34, 3),
        (21, 7.4, 101.0, 305.0, 0.29, 2),
        (14, 6.6, 110.0, 298.0, 0.33, 3),
    )
    now = now or datetime.now(timezone.utc).replace(microsecond=0)
    history = []
    for days_ago, pitch_variation, cadence, reaction_ms, compactness, recalled in measurements:
        created_at = now - timedelta(days=days_ago)
        history.append({
            "id": f"demo-history-{days_ago}",
            "created_at": created_at.isoformat(),
            "summary": {
                "product": "NeuroNexus",
                "version": "2.0.0",
                "created_utc": created_at.isoformat(),
                "workflow_mode": "Individual",
                "purpose": (
                    "Exploratory signal measurements only; not a diagnosis "
                    "or disease-risk estimate."
                ),
                "demo_history": True,
                "demo_mode": True,
                "synthetic_demo": {
                    "profile": "Synthetic multi-week timeline example",
                    "modules": [
                        "Vocal pitch stability",
                        "Movement / gait cadence",
                        "Reaction time",
                        "Clock drawing",
                        "Memory check",
                    ],
                    "notice": DEMO_MODE_NOTICE,
                },
                "voice_pattern_features": {"std_pitch": pitch_variation},
                "gait": {"cadence_steps_per_min": cadence},
                "clock_drawing_measurements": {
                    "bounding_box_area_fraction": compactness,
                },
                "cognitive_tasks": {
                    "reaction_time": {
                        "trials": 5,
                        "median_ms": reaction_ms,
                    },
                    "clock_drawing": {"completed": True},
                    "memory_check": {
                        "completed": True,
                        "completion_status": "completed",
                        "items_presented": 5,
                        "items_recalled": recalled,
                        "recall_percentage": recalled * 20.0,
                        "words_recalled": list(_DEMO_MEMORY_WORDS[:recalled]),
                        "words_not_recalled": list(_DEMO_MEMORY_WORDS[recalled:]),
                        "presentation_seconds": 10.0,
                        "recall_latency_seconds": 4.2 + (3 - recalled) * 0.7,
                        "completion_time_seconds": 15.0 + recalled,
                        "method": "Five-word recall after timed word presentation.",
                        "interpretation": "Descriptive prototype task only; not diagnostic.",
                    },
                },
            },
        })
    return sorted(history, key=lambda row: row["created_at"], reverse=True)


def initial_demo_sessions(now=None):
    """Return the demo timeline in the same serialized shape as persisted sessions."""
    return [
        {
            **row,
            "summary": json.dumps(row["summary"], allow_nan=False),
        }
        for row in build_synthetic_history(now)
    ]


def demo_user():
    """Return a synthetic identity that has no persistent account ID."""
    return {
        "id": None,
        "name": "Demo Participant",
        "email": "demo-participant@neuronexus.local",
        "role": "individual",
        "consent_flags": {},
        "demo_mode": True,
    }


def is_demo_mode(user):
    return isinstance(user, dict) and user.get("demo_mode") is True


def enter_demo_mode(state):
    """Enter an isolated demo session without initializing or querying the database."""
    from .session import initialize_session, reset_session

    if state.get("user") is not None:
        raise ValueError("Sign out of the current account before entering Demo Mode.")
    initialize_session(state)
    reset_session(state)
    state["user"] = demo_user()
    state["demo_mode"] = True
    state["demo_sessions"] = initial_demo_sessions()
    state["workspace_page"] = "My dashboard"


def reset_demo_mode(state):
    """Restore initial synthetic history and clear in-session demo activity."""
    if not is_demo_mode(state.get("user")):
        raise ValueError("Demo data can only be reset from an active Demo Mode session.")

    from .session import reset_session

    reset_session(state)
    for key in list(state.keys()):
        if key.startswith(("demo_", "wellness_")):
            state.pop(key, None)
    state["user"] = demo_user()
    state["demo_mode"] = True
    state["demo_sessions"] = initial_demo_sessions()
    state["workspace_page"] = "My dashboard"
    state["reaction_started"] = None
    state["reaction_ready"] = False
    state["demo_reset_notice"] = True


def sessions_for_user(user, state):
    """Read only in-memory demo sessions for Demo Mode; use storage for real users."""
    if is_demo_mode(user):
        return list(state.get("demo_sessions") or [])
    if user.get("id") is None:
        raise ValueError("A persisted account ID is required to load saved sessions.")
    from .accounts import user_sessions

    return user_sessions(user["id"])


def save_demo_report(state, report):
    """Append a visibly marked report to the current demo session, never the database."""
    if not is_demo_mode(state.get("user")):
        raise ValueError("Demo reports can only be saved in an active Demo Mode session.")
    if not isinstance(report, dict):
        raise ValueError("A demo report must be an object.")

    saved_report = dict(report)
    saved_report["demo_mode"] = True
    saved_report["demo_notice"] = DEMO_MODE_NOTICE
    created_at = datetime.now(timezone.utc).isoformat()
    saved_report.setdefault("created_utc", created_at)
    session = {
        "id": f"demo-session-{uuid4().hex[:12]}",
        "created_at": created_at,
        "summary": json.dumps(saved_report, allow_nan=False),
    }
    sessions = list(state.get("demo_sessions") or [])
    sessions.insert(0, session)
    state["demo_sessions"] = sessions
    return session["id"]
