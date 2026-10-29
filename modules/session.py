"""Session lifecycle helpers."""
RESULT_KEYS = ("audio_result", "motion_result", "gait_result")
DEFAULTS = {
    "audio_result": None,
    "motion_result": None,
    "gait_result": None,
    "upload_generation": 0,
    "consent_granted": False,
}
def initialize_session(state):
    for key, value in DEFAULTS.items():
        if key not in state:
            state[key] = value
def reset_session(state):
    for key in RESULT_KEYS:
        state[key] = None
    state["consent_granted"] = False
    state["upload_generation"] = int(state.get("upload_generation", 0)) + 1
