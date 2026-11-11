from modules.pdf_report import make_pdf
from modules.reporting import key_measurement_rows


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
        b"Clock drawing",
        b"Speech features",
        b"Research model status",
        b"Key measurement breakdown",
        b"Clinician Discussion Guide",
        b"clinically",
        b"validated",
    ):
        assert section in pdf


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
