"""Session lifecycle helpers for the NeuroNexus Streamlit prototype."""

RESULT_KEYS = ("audio_result", "motion_result")
DEFAULTS = {
    "audio_result": None,
    "motion_result": None,
    "upload_generation": 0,
}


def initialize_session(session_state) -> None:
    """Initialize app-owned state without overwriting an active session."""
    for key, value in DEFAULTS.items():
        if key not in session_state:
            session_state[key] = value


def reset_session(session_state) -> None:
    """Clear app-held results and invalidate upload/consent widget keys.

    This removes references from Streamlit session state; it cannot promise
    physical memory erasure or clear browser, host, proxy, or monitoring logs.
    """
    for key in RESULT_KEYS:
        session_state[key] = None
    session_state["upload_generation"] = int(session_state.get("upload_generation", 0)) + 1
