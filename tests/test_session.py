from modules.session import initialize_session, reset_session


def test_initialize_session_does_not_overwrite_existing_results():
    state = {"audio_result": {"duration_seconds": 3}, "upload_generation": 4}
    initialize_session(state)
    assert state["audio_result"] == {"duration_seconds": 3}
    assert state["motion_result"] is None
    assert state["upload_generation"] == 4


def test_reset_clears_results_and_increments_widget_generation():
    state = {
        "audio_result": {"duration_seconds": 3},
        "motion_result": {"sample_count": 10},
        "upload_generation": 2,
    }
    reset_session(state)
    assert state["audio_result"] is None
    assert state["motion_result"] is None
    assert state["upload_generation"] == 3
