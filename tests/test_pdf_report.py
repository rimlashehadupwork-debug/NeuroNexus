from modules.pdf_report import make_pdf
from modules.pdf_report import _research_model_rows, _table
from modules.reporting import (
    build_report,
    completed_module_explanations,
    humanize_feature_code,
    key_measurement_rows,
)
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph
from io import BytesIO


def test_missing_measurements_are_labelled_and_never_render_as_none_ms():
    report = {
        "cognitive_tasks": {
            "clock_drawing": None,
            "reaction_time": {
                "trials": 0,
                "median_ms": None,
                "variability_ms": None,
            },
        }
    }

    pdf = make_pdf("Test participant", report)
    measurements = key_measurement_rows(report)

    assert pdf.startswith(b"%PDF")
    assert b"None ms" not in pdf
    assert b"Not completed" in pdf
    assert all(row["Recorded value"] != "None" for row in measurements)


def test_pdf_includes_session_history_wellness_and_privacy_aware_metadata():
    report = {
        "workflow_mode": "Individual",
        "voice": {
            "duration_seconds": 6,
            "sample_rate": 16000,
            "source_sample_rate": 22050,
            "source_channels": 1,
            "conversion_method": "soundfile",
        },
        "voice_pattern_features": {"std_pitch": 1.2},
        "cognitive_tasks": {
            "memory_check": {
                "completed": True,
                "items_recalled": 3,
                "items_presented": 5,
                "recall_percentage": 60,
            },
        },
    }
    history = [
        {
            "id": 12,
            "created_at": "2025-02-01T10:00:00+00:00",
            "summary": {
                "workflow_mode": "Individual",
                "voice_pattern_features": {"std_pitch": 1.4},
                "cognitive_tasks": {"memory_check": {"completed": True}},
                "demo_mode": True,
            },
        },
        {
            "id": 22,
            "created_at": "2025-03-01T10:00:00+00:00",
            "summary": {
                **report,
                "created_utc": "2025-03-01T10:00:00+00:00",
            },
        },
    ]

    pdf = make_pdf(
        "Private participant",
        report,
        session_id=22,
        session_created_at="2025-03-01T10:00:00+00:00",
        history=history,
        wellness_plan={
            "goals": ["Sleep routine"],
            "recommendations": ["Maintain a consistent sleep routine."],
        },
    )

    for section in (
        b"Session ID",
        b"22",
        b"Assessment date and time",
        b"Audio processing information",
        b"Source sample rate",
        b"Participant-specific model output",
        b"No participant-specific model output",
        b"Memory assessment",
        b"Brain-health session history",
        b"DEMO / SAMPLE",
        b"Vocal pitch stability",
        b"Wellness recommendations",
        b"Sleep routine",
        b"Page 1",
    ):
        assert section in pdf
    assert b"Private participant" not in pdf


def test_incomplete_pdf_is_explicit_and_non_finite_measurements_are_missing():
    incomplete_pdf = make_pdf("Test participant", {})
    report = {"voice": {"duration_seconds": float("nan")}}
    pdf = make_pdf("Test participant", report)
    measurements = key_measurement_rows(report)

    assert b"Incomplete" in incomplete_pdf
    assert b"No completed assessment modules" in incomplete_pdf
    assert all(row["Recorded value"] != "nan" for row in measurements)
    assert b"(nan)" not in pdf.lower()


def test_pdf_includes_assessment_status_and_non_diagnostic_guide():
    report = {
        "clock_drawing_measurements": {"stroke_count": 4},
        "cognitive_tasks": {
            "clock_drawing": {"completed": True},
            "reaction_time": {"trials": 2, "median_ms": 310, "variability_ms": 12},
        },
        "alzheimers_speech_features": {"n_words": 20},
    }
    model_status = {
        "parkinsons": {
            "available": True,
            "subject_level": {"balanced_accuracy": 0.62},
        },
        "alzheimers": {
            "available": False,
            "message": "No compatible model is installed.",
        },
    }

    pdf = make_pdf("Test participant", report, model_status)

    for section in (
        b"How to Read Your Results",
        b"Stroke count, path length, and the portion of the canvas used in your drawing.",
        b"Clock drawing",
        b"Speech features",
        b"Research model status",
        b"Key measurement breakdown",
        b"Clinician Discussion Guide",
        b"clinically",
        b"validated",
    ):
        assert section in pdf


def test_pdf_includes_available_experimental_voice_model_output():
    pdf = make_pdf(
        "Test participant",
        {
            "voice_pattern_features": {"std_pitch": 1.2},
            "model_outputs": {
                "sustained_vowel_research_pattern": {
                    "experimental_output": {
                        "pattern_label": "patterns closer to dataset controls",
                        "model_score": 0.437,
                        "threshold": 0.5,
                    },
                    "model_context": {
                        "subject_level_balanced_accuracy": 0.62,
                    },
                    "required_disclaimer": "Experimental research output only.",
                },
            },
        },
    )

    assert b"Participant-specific model output" in pdf
    assert b"patterns closer to dataset controls" in pdf
    assert b"0.437" in pdf
    assert b"Experimental research output only." in pdf


def test_research_status_validation_text_wraps_within_page_width():
    model_status = {
        "parkinsons": {
            "available": False,
            "message": "A detailed validation status " * 18,
        },
        "alzheimers": {
            "available": True,
            "subject_level": {"balanced_accuracy": 0.62},
        },
    }
    widths = [112, 125, 291]
    table = _table(_research_model_rows(model_status), widths, font_size=7.5)
    canvas = Canvas(BytesIO())

    table_width, _ = table.wrapOn(canvas, 528, 720)

    assert table_width <= 528
    paragraph = table._cellvalues[1][2][0]
    assert isinstance(paragraph, Paragraph)
    _, paragraph_height = paragraph.wrap(widths[2] - 10, 720)
    assert paragraph_height > 9.5


def test_completed_module_explanations_only_include_recorded_assessments():
    explanations = completed_module_explanations({
        "voice": {"duration_seconds": 5},
        "cognitive_tasks": {"reaction_time": {"trials": 0}},
    })

    assert list(explanations) == ["Voice"]
    assert "estimated vocal pitch" in explanations["Voice"]


def test_audio_source_metadata_and_descriptive_features_reach_summary_rows():
    report = build_report(audio_result={
        "duration_seconds": 6.0,
        "sample_rate": 16000,
        "source_sample_rate": 22050,
        "source_channels": 2,
        "source_duration_seconds": 6.0,
        "conversion_method": "soundfile",
        "peak_amplitude": 0.4,
        "rms": 0.2,
        "clipping_ratio": 0.0,
        "silence_ratio": 0.1,
        "zero_crossing_rate": 0.03,
        "spectral_centroid_hz": 210.0,
        "rms_variability": 0.01,
    })

    voice = report["voice"]
    assert voice["source_sample_rate"] == 22050
    assert voice["source_channels"] == 2
    measurements = {
        row["Measurement"]: row["Recorded value"]
        for row in key_measurement_rows(report)
    }
    assert measurements["Source sample rate"] == "22050 Hz"
    assert measurements["Source channels"] == "2 channels"
    assert measurements["Peak amplitude"] == "0.4"


def test_dataset_feature_codes_use_plain_language_labels():
    assert humanize_feature_code("gmrt_in_air7") == "Pen-air movement timing (task 7)"
    assert humanize_feature_code("disp_index23") == "Drawing displacement index (task 23)"


def test_pdf_clearly_discloses_synthetic_demo_values():
    report = {
        "synthetic_demo": {
            "profile": "Clean signal demo",
            "modules": ["Voice", "Clock drawing"],
            "notice": (
                "Synthetic demonstration values only. These values were not collected "
                "from a person and no trained model inference was run on them."
            ),
        }
    }

    pdf = make_pdf("Test participant", report)

    assert b"Synthetic demo data" in pdf
    assert b"not collected from a person" in pdf
    assert b"Demo-filled modules: Voice, Clock drawing" in pdf


def test_pdf_includes_optional_participant_context_and_consented_caregiver_notes():
    report = {
        "participant_context": {
            "age_bracket": "60–69",
            "biological_sex": "Prefer not to say",
            "dominant_hand": "Left",
            "primary_input_device": "Touchscreen / Stylus",
        },
        "caregiver_observations": ["Observed tremor", "Observed gait changes"],
        "assisted_session_mode": True,
    }

    pdf = make_pdf("Test participant", report)

    for section in (
        b"Participant context",
        b"60",
        b"Biological sex",
        b"Caregiver observation notes",
        b"Participant-approved, user-reported observations",
        b"Observed tremor",
        b"Observed gait changes",
        b"Assisted session mode: On",
    ):
        assert section in pdf


def test_pdf_discloses_darwin_schema_incompatibility_and_aggregate_example():
    report = {
        "cognitive_tasks": {
            "clock_drawing": {
                "completed": True,
                "darwin_model_compatibility": {
                    "mapped_feature_count": 0,
                    "required_feature_count": 20,
                    "message": "No individual prediction was calculated.",
                },
            }
        },
        "darwin_sample_profile": {
            "profile_name": "DARWIN cohort median feature profile",
            "features": {"air_time1": 420.0},
            "notice": "This is a dataset-wide median, not an individual.",
        },
    }

    pdf = make_pdf("Test participant", report)

    for section in (
        b"Experimental Kinematic Similarity Indicator",
        b"Not calculated",
        b"0 of 20 selected DARWIN features",
        b"DARWIN research example profile",
        b"cohort median",
        b"not an individual",
    ):
        assert section in pdf


def test_memory_check_metrics_appear_in_summary_measurements_and_clinician_pdf():
    report = {
        "cognitive_tasks": {
            "memory_check": {
                "completed": True,
                "completion_status": "completed",
                "items_presented": 5,
                "items_recalled": 2,
                "recall_percentage": 40,
                "words_recalled": ["APPLE", "BLUE"],
                "words_not_recalled": ["RIVER", "CHAIR", "BOOK"],
                "recall_latency_seconds": 4.8,
                "completion_time_seconds": 14.8,
            },
        },
    }

    pdf = make_pdf("Test participant", report)
    measurements = key_measurement_rows(report)

    assert b"Memory check" in pdf
    assert b"2 items" in pdf
    assert b"4.8 s" in pdf
    assert b"40 %" in pdf
    assert b"APPLE, BLUE" in pdf
    assert b"14.8 s" in pdf
    assert any(row["Measurement"] == "Items recalled" for row in measurements)
    assert any(row["Measurement"] == "Recall percentage" for row in measurements)
    assert any(row["Measurement"] == "Words not recalled" for row in measurements)
    assert any(row["Measurement"] == "Recall latency" for row in measurements)
