"""Short, non-diagnostic timed word-recall activity."""
from __future__ import annotations

from datetime import datetime, timezone
import math
import random
import re
import time


MEMORY_WORDS = ("APPLE", "RIVER", "CHAIR", "BLUE", "BOOK")
PRESENTATION_SECONDS_OPTIONS = (5, 10, 15, 20)
DEFAULT_PRESENTATION_SECONDS = 10
WORD_COUNT = 5

_ACTIVE_KEYS = (
    "memory_check_targets",
    "memory_check_started",
    "memory_check_presentation_started",
    "memory_check_presentation_seconds",
    "memory_check_recall_started",
)


def score_recall(target_words, response):
    """Count distinct presented words recalled, regardless of case or punctuation."""
    targets = {str(word).strip().casefold() for word in target_words}
    answers = set(re.findall(r"[a-z0-9]+", str(response).casefold()))
    return len(targets & answers)


def build_memory_result(
    target_words,
    response,
    recall_latency_seconds,
    completed_at=None,
    completion_time_seconds=None,
    presentation_seconds=DEFAULT_PRESENTATION_SECONDS,
):
    """Build descriptive task output without retaining the entered response."""
    if not isinstance(target_words, (list, tuple)) or not target_words:
        raise ValueError("At least one presented word is required.")
    normalized_targets = [str(word).strip().upper() for word in target_words]
    if any(not word for word in normalized_targets):
        raise ValueError("Presented words must not be empty.")
    if len({word.casefold() for word in normalized_targets}) != len(normalized_targets):
        raise ValueError("Presented words must be unique.")

    def valid_duration(value):
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and value >= 0
        )

    if not valid_duration(recall_latency_seconds):
        raise ValueError("Recall latency must be finite and non-negative.")
    if completion_time_seconds is None:
        completion_time_seconds = recall_latency_seconds
    if not valid_duration(completion_time_seconds):
        raise ValueError("Completion time must be finite and non-negative.")
    if (
        isinstance(presentation_seconds, bool)
        or not isinstance(presentation_seconds, (int, float))
        or not math.isfinite(presentation_seconds)
        or presentation_seconds <= 0
    ):
        raise ValueError("Presentation time must be finite and greater than zero.")

    answer_tokens = set(re.findall(r"[a-z0-9]+", str(response).casefold()))
    recalled = [word for word in normalized_targets if word.casefold() in answer_tokens]
    not_recalled = [
        word for word in normalized_targets if word.casefold() not in answer_tokens
    ]
    items_recalled = len(recalled)
    return {
        "completed": True,
        "completion_status": "completed",
        "items_presented": len(normalized_targets),
        "items_recalled": items_recalled,
        "recall_percentage": round(100.0 * items_recalled / len(normalized_targets), 1),
        "words_recalled": recalled,
        "words_not_recalled": not_recalled,
        "presentation_seconds": round(float(presentation_seconds), 1),
        "recall_latency_seconds": round(float(recall_latency_seconds), 1),
        "completion_time_seconds": round(float(completion_time_seconds), 1),
        "completed_at": completed_at or datetime.now(timezone.utc).isoformat(),
        "method": "Five-word recall after timed word presentation.",
        "interpretation": (
            "Descriptive prototype task only; not a validated clinical assessment "
            "or a diagnosis."
        ),
    }


def _exit_to_instructions(state):
    state["memory_check_phase"] = "ready"
    for key in _ACTIVE_KEYS:
        state.pop(key, None)


def _render_results(state, st, on_change):
    result = state.get("memory_result") or {}
    if not result.get("completed"):
        st.warning("No completed memory result is available.")
        return

    st.progress(1.0)
    st.caption("Step 4 of 4 · Results")
    st.success("Thanks for taking part. Your task result is ready.")
    st.info(
        "This is a prototype memory task, not a validated clinical test. "
        "The result does not diagnose Alzheimer's disease or any other condition."
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric("Words presented", result["items_presented"])
    metric_cols[1].metric(
        "Correct words recalled",
        f"{result['items_recalled']} of {result['items_presented']}",
    )
    percentage = result.get("recall_percentage")
    metric_cols[2].metric(
        "Recall percentage",
        f"{percentage:.0f}%" if isinstance(percentage, (int, float)) else "Unavailable",
    )
    completion_time = result.get("completion_time_seconds")
    metric_cols[3].metric(
        "Completion time",
        f"{completion_time:.1f} seconds"
        if isinstance(completion_time, (int, float))
        else "Unavailable",
    )

    recalled = result.get("words_recalled")
    not_recalled = result.get("words_not_recalled")
    recalled_text = (
        ", ".join(recalled) if recalled else "No words were recalled."
    ) if isinstance(recalled, list) else "Unavailable for this saved result."
    not_recalled_text = (
        ", ".join(not_recalled) if not_recalled else "None."
    ) if isinstance(not_recalled, list) else "Unavailable for this saved result."
    recalled_col, missed_col = st.columns(2)
    recalled_col.markdown("**Words successfully recalled**")
    recalled_col.write(recalled_text)
    missed_col.markdown("**Words not recalled**")
    missed_col.write(not_recalled_text)
    presentation_seconds = result.get("presentation_seconds")
    recall_latency = result.get("recall_latency_seconds")
    if presentation_seconds is not None or recall_latency is not None:
        display_text = (
            f"Word display: {presentation_seconds:.1f} seconds"
            if isinstance(presentation_seconds, (int, float))
            else "Word display time unavailable"
        )
        recall_text = (
            f"Recall entry time: {recall_latency:.1f} seconds"
            if isinstance(recall_latency, (int, float))
            else "Recall entry time unavailable"
        )
        st.caption(
            f"{display_text} · {recall_text}. "
            "No comparison with a clinical reference range is made."
        )

    restart_col, exit_col = st.columns(2)
    if restart_col.button(
        "Restart memory assessment",
        key="memory_check_restart",
        type="primary",
    ):
        state["memory_result"] = None
        _exit_to_instructions(state)
        on_change()
        st.rerun()
    if exit_col.button("Exit memory check", key="memory_check_exit_results"):
        _exit_to_instructions(state)
        on_change()
        st.rerun()


def render_memory_check(state, on_change):
    """Render instructions, timed word presentation, recall, and results."""
    import streamlit as st

    phase = state.get("memory_check_phase", "ready")

    if phase == "ready":
        st.progress(0.0)
        st.caption("Step 1 of 4 · Instructions")
        if (state.get("memory_result") or {}).get("completed"):
            st.info(
                "Your previous result is still included in this session summary. "
                "Starting again replaces it with a new task result."
            )
        with st.container(border=True):
            st.subheader("Prototype memory task")
            st.write(
                "You will see a few words. Try to remember them. After they "
                "disappear, you will be asked to recall as many as you can."
            )
            st.caption(
                "This short activity is for demonstration and cognitive "
                "assessment research only. It is not a validated clinical test "
                "and does not diagnose Alzheimer's disease or any other condition."
            )
            duration_selection = state.get(
                "memory_check_duration_selection",
                DEFAULT_PRESENTATION_SECONDS,
            )
            if duration_selection not in PRESENTATION_SECONDS_OPTIONS:
                duration_selection = DEFAULT_PRESENTATION_SECONDS
            presentation_seconds = st.selectbox(
                "Word display time",
                PRESENTATION_SECONDS_OPTIONS,
                index=PRESENTATION_SECONDS_OPTIONS.index(duration_selection),
                format_func=lambda value: f"{value} seconds",
                key="memory_check_duration_selection",
            )
            st.checkbox(
                "Show countdown while words are visible",
                value=state.get("memory_check_show_timer", True),
                key="memory_check_show_timer",
            )
            if st.button(
                "Start memory assessment",
                key="memory_check_start",
                type="primary",
            ):
                now = time.monotonic()
                state["memory_check_targets"] = random.sample(
                    MEMORY_WORDS,
                    WORD_COUNT,
                )
                state["memory_check_started"] = now
                state["memory_check_presentation_started"] = now
                state["memory_check_presentation_seconds"] = presentation_seconds
                state["memory_check_phase"] = "presentation"
                state["memory_result"] = None
                on_change()
                st.rerun()

    elif phase == "presentation":
        @st.fragment(run_every=0.5, key="memory-word-presentation")
        def render_timed_words():
            if state.get("memory_check_phase") != "presentation":
                return

            elapsed = max(
                0.0,
                time.monotonic() - state["memory_check_presentation_started"],
            )
            duration = float(state["memory_check_presentation_seconds"])
            if elapsed >= duration:
                state["memory_check_phase"] = "recall"
                state["memory_check_recall_started"] = time.monotonic()
                on_change()
                st.rerun()

            st.progress(0.35)
            st.caption("Step 2 of 4 · Remember these words")
            st.subheader(" · ".join(state["memory_check_targets"]))
            st.caption(
                "The words will disappear automatically after the selected display time."
            )
            if state.get("memory_check_show_timer", True):
                remaining = max(0, math.ceil(duration - elapsed))
                st.caption(f"Words will be hidden in about {remaining} seconds.")
            if st.button(
                "Exit memory check",
                key="memory_check_exit_presentation",
            ):
                _exit_to_instructions(state)
                on_change()
                st.rerun()

        render_timed_words()

    elif phase == "recall":
        st.progress(0.7)
        st.caption("Step 3 of 4 · Recall")
        with st.container(border=True):
            st.subheader("Which words do you remember?")
            st.write("Enter as many words as you can recall, in any order.")
            with st.form("memory_check_recall_form"):
                response = st.text_input(
                    "Words you remember",
                    max_chars=120,
                    placeholder="Type words separated by spaces or commas",
                    help="Your full text response is not included in the saved result.",
                    key="memory_check_response",
                )
                submitted = st.form_submit_button(
                    "Submit answers",
                    type="primary",
                    width="stretch",
                )
            if submitted:
                now = time.monotonic()
                recall_latency = max(
                    0.0,
                    now - state["memory_check_recall_started"],
                )
                completion_time = max(
                    0.0,
                    now - state["memory_check_started"],
                )
                state["memory_result"] = build_memory_result(
                    state["memory_check_targets"],
                    response,
                    recall_latency,
                    completion_time_seconds=completion_time,
                    presentation_seconds=state["memory_check_presentation_seconds"],
                )
                state["memory_check_phase"] = "complete"
                for key in _ACTIVE_KEYS:
                    state.pop(key, None)
                on_change()
                st.rerun()
            if st.button("Exit memory check", key="memory_check_exit_recall"):
                _exit_to_instructions(state)
                on_change()
                st.rerun()

    elif phase == "complete":
        _render_results(state, st, on_change)
    else:
        raise ValueError(f"Unknown memory assessment phase: {phase}")
