from modules import accounts
from modules.session import (
    assessment_completion,
    assessment_draft_payload,
    has_assessment_progress,
    initialize_session,
    reset_session,
    restore_assessment_draft,
)


def test_initialize_and_reset():
    state = {}
    initialize_session(state)
    state["audio_result"] = {"x": 1}
    state["clock_result"] = {"features": {"stroke_count": 1}}
    state["pattern_result"] = {"experimental_output": {"pattern_label": "test"}}
    state["voice_wav_bytes"] = b"raw audio"
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
    assert state["baseline_context"] is None
    assert state["caregiver_observation_notes"] == []
    assert state["caregiver_notes_consent"] is False
    assert state["assisted_session_mode"] is False
    assert state["local_history_enabled"] is False
    assert state["consent_granted"] is False
    assert state["upload_generation"] == old + 1


def test_resume_restores_completed_modules_without_raw_audio_and_reset_purges_draft(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(accounts, "DB_PATH", tmp_path / "accounts.sqlite3")
    accounts.init_db()
    accounts.register("Participant", "resume@example.test", "long-password")
    user = accounts.authenticate("resume@example.test", "long-password")

    source = {}
    initialize_session(source)
    source["audio_result"] = {"duration_seconds": 5.0}
    source["reaction_times"] = [301.0]
    source["voice_wav_bytes"] = b"raw audio"
    payload = assessment_draft_payload(source)
    assert has_assessment_progress(source)
    assert "voice_wav_bytes" not in payload
    accounts.save_draft(user["id"], payload)

    resumed = {"user": user}
    initialize_session(resumed)
    restore_assessment_draft(resumed, accounts.load_draft(user["id"]))
    assert resumed["audio_result"] == {"duration_seconds": 5.0}
    assert resumed["reaction_times"] == [301.0]
    assert resumed["voice_wav_bytes"] is None

    reset_session(resumed)
    assert accounts.load_draft(user["id"]) is None


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
