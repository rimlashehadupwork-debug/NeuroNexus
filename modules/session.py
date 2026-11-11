"""Session lifecycle helpers."""
RESULT_KEYS = (
    "audio_result",
    "motion_result",
    "gait_result",
    "alz_result",
    "clock_result",
    "pattern_result",
    "voice_wav_bytes",
    "demo_profile",
    "reaction_times_demo",
)
DEFAULTS = {
    "audio_result": None,
    "motion_result": None,
    "gait_result": None,
    "alz_result": None,
    "clock_result": None,
    "pattern_result": None,
    "voice_wav_bytes": None,
    "demo_profile": None,
    "reaction_times_demo": False,
    "darwin_sample_profile": None,
    "baseline_context": None,
    "caregiver_observation_notes": None,
    "caregiver_notes_consent": False,
    "assisted_session_mode": False,
    "local_history_enabled": False,
    "profile_widget_generation": 0,
    "upload_generation": 0,
    "consent_granted": False,
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


def reset_session(state):
    for key in RESULT_KEYS:
        state[key] = None
    state["demo_profile"] = None
    state["reaction_times_demo"] = False
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
    state["consent_granted"] = False
    state["upload_generation"] = int(state.get("upload_generation", 0)) + 1
