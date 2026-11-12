"""Session lifecycle helpers."""
RESULT_KEYS = (
    "audio_result",
    "motion_result",
    "gait_result",
    "alz_result",
    "clock_result",
    "pattern_result",
    "voice_wav_bytes",
)
DEFAULTS = {
    "audio_result": None,
    "motion_result": None,
    "gait_result": None,
    "alz_result": None,
    "clock_result": None,
    "pattern_result": None,
    "voice_wav_bytes": None,
    "darwin_sample_profile": None,
    "baseline_context": None,
    "caregiver_observation_notes": None,
    "caregiver_notes_consent": False,
    "assisted_session_mode": False,
    "local_history_enabled": False,
    "profile_widget_generation": 0,
    "upload_generation": 0,
    "consent_granted": False,
    "draft_resumed": False,
    "draft_finalized": False,
}


ASSESSMENT_MODULES = (
    "voice",
    "resting_motion",
    "gait",
    "clock_drawing",
    "reaction_time",
    "speech_features",
    "darwin_sample_profile",
)


def assessment_completion(state):
    """Return completion status for the six assessable modules (not the report tab)."""
    reaction_times = state.get("reaction_times") or []
    return {
        "voice": bool(state.get("audio_result")),
        "resting_motion": bool(state.get("motion_result")),
        "gait": bool(state.get("gait_result")),
        "clock_drawing": bool(
            (state.get("clock_result") or {}).get("completed")
        ),
        "reaction_time": len(reaction_times) >= 5,
        "speech_features": bool(state.get("alz_result")),
    }


def initialize_session(state):
    for key, value in DEFAULTS.items():
        if key not in state:
            state[key] = value


def assessment_draft_payload(state):
    """Return resumable module results without raw audio or upload contents."""
    return {
        "audio_result": state.get("audio_result"),
        "motion_result": state.get("motion_result"),
        "gait_result": state.get("gait_result"),
        "alz_result": state.get("alz_result"),
        "clock_result": state.get("clock_result"),
        "pattern_result": state.get("pattern_result"),
        "reaction_times": list(state.get("reaction_times") or []),
    }


def has_assessment_progress(state):
    return any((
        state.get("audio_result"),
        state.get("motion_result"),
        state.get("gait_result"),
        state.get("alz_result"),
        state.get("clock_result"),
        state.get("pattern_result"),
        state.get("reaction_times"),
    ))


def restore_assessment_draft(state, draft):
    if not isinstance(draft, dict):
        raise ValueError("The saved assessment draft must be an object.")
    for key in (
        "audio_result",
        "motion_result",
        "gait_result",
        "alz_result",
        "clock_result",
        "pattern_result",
    ):
        state[key] = draft.get(key)
    reaction_times = draft.get("reaction_times") or []
    if not isinstance(reaction_times, list):
        raise ValueError("Saved reaction-time trials must be a list.")
    state["reaction_times"] = reaction_times
    state["voice_wav_bytes"] = None
    state["draft_finalized"] = False
    state["upload_generation"] = int(state.get("upload_generation", 0)) + 1


def reset_session(state, reset_consent=True):
    user = state.get("user")
    if user and user.get("id") is not None:
        from .accounts import delete_draft

        delete_draft(user["id"])
    for key in RESULT_KEYS:
        state[key] = None
    state["darwin_sample_profile"] = None
    state["baseline_context"] = None
    state["caregiver_observation_notes"] = []
    state["caregiver_notes_consent"] = False
    state["assisted_session_mode"] = False
    state["local_history_enabled"] = False
    state["profile_widget_generation"] = int(
        state.get("profile_widget_generation", 0)
    ) + 1
    state["profile_notice"] = None
    if reset_consent:
        state["consent_granted"] = False
    state["draft_resumed"] = False
    state["draft_finalized"] = False
    state["reaction_times"] = []
    state["upload_generation"] = int(state.get("upload_generation", 0)) + 1
