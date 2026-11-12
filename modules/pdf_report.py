"""Generate non-diagnostic, missing-value-safe screening session PDFs."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from html import escape
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .reporting import completed_module_explanations, humanize_feature_code, key_measurement_rows


NOT_COMPLETED = "Not completed"


def _display_value(value):
    if value is None or value == "":
        return NOT_COMPLETED
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.4g}" if math.isfinite(value) else NOT_COMPLETED
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    return str(value)


def _paragraph(text, style):
    return Paragraph(escape(_display_value(text)), style)


def _table(rows, widths, font_size=8):
    body_style = ParagraphStyle(
        "TableBody",
        fontName="Helvetica",
        fontSize=font_size,
        leading=font_size + 2,
    )
    header_style = ParagraphStyle(
        "TableHeader",
        parent=body_style,
        fontName="Helvetica-Bold",
        textColor=colors.white,
    )
    paragraph_rows = [
        [
            Paragraph(escape(_display_value(cell)), header_style if row_index == 0 else body_style)
            for cell in row
        ]
        for row_index, row in enumerate(rows)
    ]
    table = Table(paragraph_rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d192b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dbe3ed")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def _how_to_read_box(report, styles, width):
    explanations = completed_module_explanations(report)
    rows = [[Paragraph("How to Read Your Results", styles["Heading3"])]]
    if explanations:
        rows.extend(
            [[Paragraph(f"<b>{escape(label)}:</b> {escape(description)}", styles["BodyText"])]]
            for label, description in explanations.items()
        )
    else:
        rows.append([
            Paragraph(
                "No completed assessment modules are recorded in this session.",
                styles["BodyText"],
            )
        ])
    box = Table(rows, colWidths=[width], hAlign="LEFT")
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#edf5f2")),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#c6d8d2")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return box


def _research_model_rows(model_status):
    rows = [["Pathway", "Availability", "Validation context"]]
    for key, label in (
        ("parkinsons", "Parkinson's voice"),
        ("alzheimers", "Alzheimer's kinematics (DARWIN)"),
    ):
        status = (model_status or {}).get(key) or {}
        if not status.get("available"):
            detail = status.get("message") or "No compatible trained model is installed."
            rows.append([label, "Unavailable", _display_value(detail)])
            continue

        subject = status.get("subject_level") or {}
        metrics = []
        for metric, metric_label in (
            ("accuracy", "accuracy"),
            ("balanced_accuracy", "balanced accuracy"),
            ("sensitivity", "sensitivity"),
            ("specificity", "specificity"),
            ("roc_auc", "ROC-AUC"),
        ):
            value = subject.get(metric)
            if value is not None:
                metrics.append(f"{metric_label} {_display_value(value)}")
        validation = ", ".join(metrics) or "No validation metrics reported"
        detail = (
            f"Internal grouped-CV only ({validation}); not externally or clinically "
            "validated for individual screening."
        )
        rows.append([label, "Research artifact installed", detail])
    return rows


def make_pdf(name, report, model_status=None):
    report = report or {}
    out = BytesIO()
    doc = SimpleDocTemplate(
        out,
        pagesize=letter,
        rightMargin=42,
        leftMargin=42,
        topMargin=42,
        bottomMargin=42,
        pageCompression=0,
    )
    styles = getSampleStyleSheet()
    story = [
        _paragraph("NeuroNexus — Screening Session Summary", styles["Title"]),
        Spacer(1, 8),
        _paragraph(f"Participant: {name}", styles["Normal"]),
        _paragraph(f"Generated: {report.get('created_utc')}", styles["Normal"]),
        Spacer(1, 10),
        _how_to_read_box(report, styles, doc.width),
        Spacer(1, 14),
        _paragraph(
            "Important: This is an experimental research summary, not a diagnosis "
            "or disease-risk estimate. It cannot confirm or rule out Alzheimer's "
            "disease or Parkinson's disease. Discuss symptoms or concerns with a "
            "qualified clinician.",
            styles["BodyText"],
        ),
        Spacer(1, 14),
    ]

    synthetic_demo = report.get("synthetic_demo")
    if synthetic_demo:
        modules = ", ".join(synthetic_demo.get("modules") or []) or NOT_COMPLETED
        story.extend([
            _paragraph("Synthetic demo data", styles["Heading2"]),
            _paragraph(synthetic_demo.get("profile"), styles["BodyText"]),
            _paragraph(synthetic_demo.get("notice"), styles["BodyText"]),
            _paragraph(f"Demo-filled modules: {modules}", styles["BodyText"]),
            Spacer(1, 10),
        ])

    participant_context = report.get("participant_context") or {}
    if participant_context:
        context_rows = [["Participant context", "Self-reported value"]]
        for key, label in (
            ("age_bracket", "Age bracket"),
            ("biological_sex", "Biological sex"),
            ("dominant_hand", "Dominant hand"),
            ("primary_input_device", "Primary input device"),
        ):
            if participant_context.get(key):
                context_rows.append([label, _display_value(participant_context[key])])
        if len(context_rows) > 1:
            story.extend([
                _paragraph("Participant context", styles["Heading2"]),
                _table(context_rows, [180, 348]),
                _paragraph(
                    "Context is optional and self-reported. It is not a validated "
                    "clinical adjustment or risk factor in this report.",
                    styles["BodyText"],
                ),
                Spacer(1, 10),
            ])

    caregiver_observations = report.get("caregiver_observations") or []
    if caregiver_observations:
        story.extend([
            _paragraph("Caregiver observation notes", styles["Heading2"]),
            _paragraph(
                "Participant-approved, user-reported observations; not verified "
                "clinical findings or model outputs.",
                styles["BodyText"],
            ),
        ])
        for observation in caregiver_observations:
            story.append(_paragraph(f"- {observation}", styles["BodyText"]))
        if report.get("assisted_session_mode"):
            story.append(_paragraph("Assisted session mode: On", styles["BodyText"]))
        story.append(Spacer(1, 10))

    darwin_sample = report.get("darwin_sample_profile") or {}
    if darwin_sample:
        story.extend([
            _paragraph("DARWIN research example profile", styles["Heading2"]),
            _paragraph(darwin_sample.get("profile_name"), styles["BodyText"]),
            _paragraph(darwin_sample.get("notice"), styles["BodyText"]),
        ])
        sample_rows = [["Selected feature", "Cohort median"]]
        for feature, value in (darwin_sample.get("features") or {}).items():
            sample_rows.append([humanize_feature_code(feature), _display_value(value)])
        if len(sample_rows) > 1:
            story.extend([
                _table(sample_rows, [300, 228], font_size=7.5),
                Spacer(1, 10),
            ])

    cognitive = report.get("cognitive_tasks") or {}
    clock = cognitive.get("clock_drawing") or {}
    speech = report.get("alzheimers_speech_features") or {}
    reaction = cognitive.get("reaction_time") or {}
    reaction_completed = bool(reaction.get("trials", 0))
    assessment_rows = [["Assessment", "Status", "Technical quality / note"]]
    for label, key in (
        ("Voice", "voice"),
        ("Resting motion", "resting_motion"),
        ("Gait", "gait"),
    ):
        item = report.get(key) or {}
        quality = item.get("quality") or {}
        quality_score = quality.get("score")
        quality_text = (
            f"{_display_value(quality_score)}/100"
            if quality_score is not None
            else "Not available"
        )
        assessment_rows.append([
            label,
            "Completed" if item else NOT_COMPLETED,
            quality_text if item else NOT_COMPLETED,
        ])
    assessment_rows.extend([
        [
            "Clock drawing",
            "Completed" if clock.get("completed") else NOT_COMPLETED,
            "No clinical score calculated" if clock.get("completed") else NOT_COMPLETED,
        ],
        [
            "Speech features",
            "Completed" if speech else NOT_COMPLETED,
            "Descriptive measurements only" if speech else NOT_COMPLETED,
        ],
        [
            "Reaction time",
            "Completed" if reaction_completed else NOT_COMPLETED,
            "Browser/device timing is approximate" if reaction_completed else NOT_COMPLETED,
        ],
    ])
    story.extend([
        _paragraph("Assessment summary", styles["Heading2"]),
        _table(assessment_rows, [115, 100, 313]),
        Spacer(1, 14),
    ])

    darwin_compatibility = (clock or {}).get("darwin_model_compatibility")
    if darwin_compatibility:
        story.extend([
            _paragraph("Experimental Kinematic Similarity Indicator", styles["Heading2"]),
            _paragraph("Not calculated", styles["BodyText"]),
            _paragraph(darwin_compatibility.get("message"), styles["BodyText"]),
            _paragraph(
                f"Feature schema overlap: "
                f"{_display_value(darwin_compatibility.get('mapped_feature_count'))} of "
                f"{_display_value(darwin_compatibility.get('required_feature_count'))} selected DARWIN features.",
                styles["BodyText"],
            ),
            Spacer(1, 10),
        ])

    if model_status:
        story.extend([
            _paragraph("Research model status", styles["Heading2"]),
            _table(_research_model_rows(model_status), [112, 125, 291], font_size=7.5),
            Spacer(1, 14),
            _paragraph(
                "Model validation metrics describe research evaluation only. "
                "They are not personal predictions and are not validated clinical "
                "reference ranges.",
                styles["BodyText"],
            ),
            Spacer(1, 8),
        ])

    story.extend([
        _paragraph("Key measurement breakdown", styles["Heading2"]),
        _paragraph(
            "Recorded values are descriptive. No validated healthy reference "
            "intervals or individual risk flags are available for these measurements.",
            styles["BodyText"],
        ),
        Spacer(1, 6),
    ])
    measurements = key_measurement_rows(report)
    measurement_table = [[
        "Domain",
        "Measurement",
        "Recorded value",
        "Reference",
        "Interpretation",
    ]]
    for row in measurements:
        measurement_table.append([
            _display_value(row["Domain"]),
            _display_value(row["Measurement"]),
            _display_value(row["Recorded value"]),
            _display_value(row["Reference range"]),
            _display_value(row["Interpretation"]),
        ])
    story.extend([
        _table(measurement_table, [72, 120, 100, 112, 124], font_size=7),
        Spacer(1, 14),
    ])

    story.append(_paragraph("Recorded signal details", styles["Heading2"]))
    for key in ("voice", "resting_motion", "gait"):
        item = report.get(key) or {}
        if not item:
            continue
        story.append(_paragraph(key.replace("_", " ").title(), styles["Heading3"]))
        for metric, value in item.items():
            if metric in ("quality", "mfcc_means", "mfcc_stds") or isinstance(value, (dict, list, tuple)):
                continue
            story.append(_paragraph(
                f"{metric.replace('_', ' ').title()}: {_display_value(value)}",
                styles["BodyText"],
            ))
        quality = item.get("quality") or {}
        quality_label = _display_value(quality.get("label"))
        quality_score = _display_value(quality.get("score"))
        story.append(_paragraph(
            f"Signal quality: {quality_label} ({quality_score}/100). "
            "This describes recording quality only.",
            styles["BodyText"],
        ))
        story.append(Spacer(1, 6))

    story.append(_paragraph("Clinician Discussion Guide", styles["Heading2"]))
    movement_recorded = bool(report.get("resting_motion") or report.get("gait"))
    cognition_recorded = bool(
        clock.get("completed")
        or reaction_completed
        or report.get("clock_drawing_measurements")
    )
    domain_rows = [
        ["Domain", "Recorded activity"],
        ["Voice / speech", "Recorded" if report.get("voice") or speech else NOT_COMPLETED],
        ["Movement", "Recorded" if movement_recorded else NOT_COMPLETED],
        ["Visuospatial / cognitive", "Recorded" if cognition_recorded else NOT_COMPLETED],
    ]
    story.extend([
        _table(domain_rows, [180, 348]),
        Spacer(1, 6),
        _paragraph(
            "This guide reports task coverage only; NeuroNexus does not determine "
            "whether a domain is normal or abnormal, identify irregularities, or "
            "assign disease risk. Share persistent concerns, when they began, and "
            "their effect on daily activities with a qualified clinician.",
            styles["BodyText"],
        ),
        Spacer(1, 10),
        _paragraph(
            "Seek urgent medical care for sudden new neurological symptoms.",
            styles["BodyText"],
        ),
    ])

    doc.build(story)
    return out.getvalue()
