from modules.session import assessment_completion, initialize_session, reset_session
def test_initialize_and_reset():
    state = {}
    initialize_session(state)
    state["audio_result"] = {"x": 1}
    state["clock_result"] = {"features": {"stroke_count": 1}}
    state["pattern_result"] = {"experimental_output": {"pattern_label": "test"}}
    state["voice_wav_bytes"] = b"raw audio"
    state["demo_profile"] = "Clean signal demo"
    state["reaction_times_demo"] = True
    state["baseline_context"] = {"age_bracket": "60–69"}
    state["caregiver_observation_notes"] = ["Observed tremor"]
    state["caregiver_notes_consent"] = True
    state["assisted_session_mode"] = True
    state["local_history_enabled"] = True
    state["consent_granted"] = True
    state["reaction_times"] = [300, 310, 320, 330, 340]
    old = state["upload_generation"]
    reset_session(state)
    assert state["audio_result"] is None
    assert state["clock_result"] is None
    assert state["pattern_result"] is None
    assert state["voice_wav_bytes"] is None
    assert state["demo_profile"] is None
    assert state["reaction_times_demo"] is False
    assert state["baseline_context"] is None
    assert state["caregiver_observation_notes"] == []
    assert state["caregiver_notes_consent"] is False
    assert state["assisted_session_mode"] is False
    assert state["local_history_enabled"] is False
    assert state["consent_granted"] is False
    assert state["upload_generation"] == old + 1


def test_assessment_completion_requires_all_five_reaction_trials():
    state = {
        "audio_result": {"duration_seconds": 5},
        "motion_result": {"duration_seconds": 5},
        "gait_result": {"duration_seconds": 5},
        "clock_result": {"completed": True},
        "alz_result": {"features": {"n_words": 10}},
        "reaction_times": [300, 310, 320, 330],
    }

    completion = assessment_completion(state)

    assert sum(completion.values()) == 5
    state["reaction_times"].append(340)
    assert sum(assessment_completion(state).values()) == 6
