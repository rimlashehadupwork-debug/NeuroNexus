"""Session lifecycle helpers."""
RESULT_KEYS = (
    "audio_result",
    "motion_result",
    "gait_result",
    "alz_result",
    "clock_result",
    "memory_result",
    "pattern_result",
    "voice_wav_bytes",
    "voice_visualization",
    "audio_pipeline_events",
)
DEFAULTS = {
    "audio_result": None,
    "motion_result": None,
    "gait_result": None,
    "alz_result": None,
    "clock_result": None,
    "memory_result": None,
    "pattern_result": None,
    "voice_wav_bytes": None,
    "voice_visualization": None,
    "audio_pipeline_events": [],
    "darwin_sample_profile": None,
    "baseline_context": None,
    "caregiver_observation_notes": None,
    "caregiver_notes_consent": False,
    "assisted_session_mode": False,
    "local_history_enabled": False,
    "profile_widget_generation": 0,
    "upload_generation": 0,
    "consent_granted": False,
    "assessment_consent_checked": False,
    "screening_mode": "Individual",
    "draft_resumed": False,
    "draft_finalized": False,
    "last_saved_report_session_id": None,
    "last_saved_report_created_at": None,
    "report_saved_notice": None,
}


ASSESSMENT_MODULES = (
    "voice",
    "resting_motion",
    "gait",
    "clock_drawing",
    "reaction_time",
    "memory_check",
    "speech_features",
    "darwin_sample_profile",
)


def assessment_completion(state):
    """Return completion status for the assessable modules (not the report tab)."""
    reaction_times = state.get("reaction_times") or []
    return {
        "voice": bool(state.get("audio_result")),
        "resting_motion": bool(state.get("motion_result")),
        "gait": bool(state.get("gait_result")),
        "clock_drawing": bool(
            (state.get("clock_result") or {}).get("completed")
        ),
        "memory_check": bool(
            (state.get("memory_result") or {}).get("completed")
        ),
        "reaction_time": len(reaction_times) >= 5,
        "speech_features": bool(state.get("alz_result")),
    }


def initialize_session(state):
    for key, value in DEFAULTS.items():
        if key not in state:
            state[key] = value


def assessment_draft_payload(state):
    """Return compact resumable results without source media or dense signal arrays."""
    audio_result = state.get("audio_result")
    if isinstance(audio_result, dict):
        audio_result = {
            key: value
            for key, value in audio_result.items()
            if key not in {"mfcc_means", "mfcc_stds"}
        }
    motion_result = state.get("motion_result")
    if isinstance(motion_result, dict):
        motion_result = {
            key: value
            for key, value in motion_result.items()
            if key not in {"frequencies_hz", "spectrum", "magnitude"}
        }
    gait_result = state.get("gait_result")
    if isinstance(gait_result, dict):
        gait_result = {
            key: value
            for key, value in gait_result.items()
            if key != "step_intervals_seconds"
        }
    return {
        "audio_result": audio_result,
        "motion_result": motion_result,
        "gait_result": gait_result,
        "alz_result": state.get("alz_result"),
        "clock_result": state.get("clock_result"),
        "memory_result": state.get("memory_result"),
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
        state.get("memory_result"),
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
        "memory_result",
        "pattern_result",
    ):
        state[key] = draft.get(key)
    state["memory_check_phase"] = (
        "complete"
        if (state.get("memory_result") or {}).get("completed")
        else "ready"
    )
    reaction_times = draft.get("reaction_times") or []
    if not isinstance(reaction_times, list):
        raise ValueError("Saved reaction-time trials must be a list.")
    state["reaction_times"] = reaction_times
    state["draft_finalized"] = False
    state["last_saved_report_session_id"] = None
    state["last_saved_report_created_at"] = None
    state["report_saved_notice"] = None
    clear_assessment_inputs(state)


def clear_assessment_inputs(state):
    """Clear unprocessed input widgets and retained audio without deleting results."""
    state["voice_wav_bytes"] = None
    state["upload_generation"] = int(state.get("upload_generation", 0)) + 1
    for key in ("demo_voice_sample", "pattern_demo_audio_bytes", "pattern_demo_active"):
        state.pop(key, None)


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
        state["assessment_consent_checked"] = False
    state["draft_resumed"] = False
    state["draft_finalized"] = False
    state["last_saved_report_session_id"] = None
    state["last_saved_report_created_at"] = None
    state["report_saved_notice"] = None
    state["reaction_times"] = []
    for key in (
        "memory_check_phase",
        "memory_check_targets",
        "memory_check_started",
        "memory_check_presentation_started",
        "memory_check_presentation_seconds",
        "memory_check_distraction_started",
        "memory_check_recall_started",
        "memory_check_distraction_seconds",
    ):
        state.pop(key, None)
    clear_assessment_inputs(state)
