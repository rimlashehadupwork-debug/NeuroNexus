from modules.pdf_report import make_pdf
from modules.pdf_report import _research_model_rows, _table
from modules.reporting import (
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
