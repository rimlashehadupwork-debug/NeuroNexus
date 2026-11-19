import pytest

from modules.memory_check import build_memory_result, score_recall


def test_recall_scoring_is_case_insensitive_order_independent_and_deduplicated():
    assert score_recall(
        ["APPLE", "RIVER", "CHAIR", "BLUE", "BOOK"],
        "chair, Apple. apple unknown",
    ) == 2


def test_memory_result_contains_descriptive_five_word_metrics_without_raw_answer():
    result = build_memory_result(
        ["APPLE", "RIVER", "CHAIR", "BLUE", "BOOK"],
        "river, book, extra",
        3.26,
        completed_at="2026-10-01T10:00:00+00:00",
        completion_time_seconds=14.26,
        presentation_seconds=10,
    )

    assert result["items_recalled"] == 2
    assert result["items_presented"] == 5
    assert result["recall_percentage"] == 40
    assert result["words_recalled"] == ["RIVER", "BOOK"]
    assert result["words_not_recalled"] == ["APPLE", "CHAIR", "BLUE"]
    assert result["recall_latency_seconds"] == 3.3
    assert result["completion_time_seconds"] == 14.3
    assert result["presentation_seconds"] == 10
    assert result["completion_status"] == "completed"
    assert "extra" not in repr(result)


@pytest.mark.parametrize("latency", [-1, float("inf"), float("nan")])
def test_memory_result_rejects_invalid_latency(latency):
    with pytest.raises(ValueError):
        build_memory_result(["amber"], "amber", latency)
