from modules import accounts
from modules.session import (
    assessment_completion,
    assessment_draft_payload,
    clear_assessment_inputs,
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
    state["memory_result"] = {"completed": True, "items_recalled": 2}
    state["pattern_result"] = {"experimental_output": {"pattern_label": "test"}}
    state["voice_wav_bytes"] = b"raw audio"
    state["baseline_context"] = {"age_bracket": "60–69"}
    state["caregiver_observation_notes"] = ["Observed tremor"]
    state["caregiver_notes_consent"] = True
    state["assisted_session_mode"] = True
    state["local_history_enabled"] = True
    state["consent_granted"] = True
    state["assessment_consent_checked"] = True
    state["screening_mode"] = "Community health worker"
    state["reaction_times"] = [300, 310, 320, 330, 340]
    old = state["upload_generation"]
    reset_session(state)
    assert state["audio_result"] is None
    assert state["clock_result"] is None
    assert state["memory_result"] is None
    assert state["pattern_result"] is None
    assert state["voice_wav_bytes"] is None
    assert state["baseline_context"] is None
    assert state["caregiver_observation_notes"] == []
    assert state["caregiver_notes_consent"] is False
    assert state["assisted_session_mode"] is False
    assert state["local_history_enabled"] is False
    assert state["consent_granted"] is False
    assert state["assessment_consent_checked"] is False
    assert state["screening_mode"] == "Community health worker"
    assert state["upload_generation"] == old + 1


def test_clear_assessment_inputs_removes_media_but_preserves_derived_results():
    state = {
        "audio_result": {"duration_seconds": 5.0},
        "voice_wav_bytes": b"converted audio",
        "demo_voice_sample": b"synthetic audio",
        "pattern_demo_audio_bytes": b"synthetic vowel",
        "pattern_demo_active": True,
        "upload_generation": 3,
    }

    clear_assessment_inputs(state)

    assert state["voice_wav_bytes"] is None
    assert state["audio_result"] == {"duration_seconds": 5.0}
    assert state["upload_generation"] == 4
    assert "demo_voice_sample" not in state
    assert "pattern_demo_audio_bytes" not in state
    assert "pattern_demo_active" not in state


def test_assessment_drafts_omit_dense_audio_and_sensor_arrays():
    state = {
        "audio_result": {
            "duration_seconds": 5.0,
            "mfcc_means": [0.1, 0.2],
            "mfcc_stds": [0.3, 0.4],
        },
        "motion_result": {
            "sample_count": 100,
            "frequencies_hz": [1.0, 2.0],
            "spectrum": [0.1, 0.2],
            "magnitude": [0.3, 0.4],
        },
        "gait_result": {
            "candidate_step_events": 6,
            "step_intervals_seconds": [0.5, 0.6],
        },
    }

    payload = assessment_draft_payload(state)

    assert payload["audio_result"] == {"duration_seconds": 5.0}
    assert payload["motion_result"] == {"sample_count": 100}
    assert payload["gait_result"] == {"candidate_step_events": 6}
    assert "mfcc_means" in state["audio_result"]
    assert "magnitude" in state["motion_result"]


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
        "memory_result": {"completed": True},
        "alz_result": {"features": {"n_words": 10}},
        "reaction_times": [300, 310, 320, 330],
    }

    completion = assessment_completion(state)

    assert sum(completion.values()) == 6
    state["reaction_times"].append(340)
    assert sum(assessment_completion(state).values()) == 7
