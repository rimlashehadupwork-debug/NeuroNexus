"""Generate non-diagnostic, missing-value-safe screening session PDFs."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from html import escape
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    Flowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .reporting import completed_module_explanations, humanize_feature_code, key_measurement_rows
from .longitudinal import (
    TIMELINE_METRICS,
    is_synthetic_session,
    session_metric_values,
    session_report,
)


NOT_COMPLETED = "Not completed"
_NAVY = colors.HexColor("#17364A")
_TEAL = colors.HexColor("#16858A")
_SAMPLE = colors.HexColor("#A96B00")
_GRID = colors.HexColor("#DCE5EC")


def _parse_datetime(value):
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_datetime(value):
    parsed = _parse_datetime(value)
    if parsed is None:
        return _display_value(value)
    return parsed.strftime("%b %d, %Y · %H:%M %Z").strip()


def _datetime_sort_key(value):
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def _history_records(history, session_id=None, limit=8):
    records = []
    for session in history or ():
        report = session_report(session)
        timestamp = session.get("created_at") or report.get("created_utc")
        records.append({
            "id": session.get("id"),
            "timestamp": timestamp,
            "date": _parse_datetime(timestamp),
            "type": report.get("workflow_mode") or "Not recorded",
            "assessments": list(completed_module_explanations(report)),
            "measurements": session_metric_values(report),
            "synthetic": is_synthetic_session(report),
        })
    records.sort(
        key=lambda item: _datetime_sort_key(item["date"])
        if item["date"] else float("-inf"),
        reverse=True,
    )
    if session_id is not None:
        selected = next(
            (item for item in records if str(item["id"]) == str(session_id)),
            None,
        )
        recent = records[:limit]
        if selected and selected not in recent:
            recent = [selected, *recent[:max(0, limit - 1)]]
            recent.sort(
                key=lambda item: _datetime_sort_key(item["date"])
                if item["date"] else float("-inf"),
                reverse=True,
            )
        return recent
    return records[:limit]


class _HistoryLineChart(Flowable):
    """Draw a small, print-friendly chart of recorded history values."""

    def __init__(self, title, points):
        super().__init__()
        self.title = title
        self.points = points
        self.height = 165

    def wrap(self, available_width, available_height):
        self.width = available_width
        return self.width, self.height

    def draw(self):
        canvas = self.canv
        left, right = 48, self.width - 14
        bottom, top = 36, self.height - 28
        chart_width, chart_height = right - left, top - bottom
        values = [point["value"] for point in self.points]
        minimum, maximum = min(values), max(values)
        spread = maximum - minimum
        padding = spread * 0.1 if spread else max(abs(minimum) * 0.1, 1.0)
        minimum -= padding
        maximum += padding

        canvas.setFillColor(_NAVY)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(0, self.height - 12, self.title)
        canvas.setStrokeColor(_GRID)
        canvas.setLineWidth(0.5)
        for index in range(4):
            ratio = index / 3
            y = bottom + ratio * chart_height
            value = minimum + ratio * (maximum - minimum)
            canvas.line(left, y, right, y)
            canvas.setFillColor(_NAVY)
            canvas.setFont("Helvetica", 7)
            canvas.drawRightString(left - 5, y - 2, f"{value:.3g}")

        dated = [point["date"] for point in self.points if point["date"]]
        ordinal_values = [value.toordinal() for value in dated]
        first_day = min(ordinal_values) if ordinal_values else 0
        last_day = max(ordinal_values) if ordinal_values else 0
        use_index = first_day == last_day
        x_values = []
        for index, point in enumerate(self.points):
            if use_index:
                ratio = index / max(1, len(self.points) - 1)
            else:
                ratio = (point["date"].toordinal() - first_day) / (last_day - first_day)
            x_values.append(left + ratio * chart_width)
        y_values = [
            bottom + (point["value"] - minimum) / (maximum - minimum) * chart_height
            for point in self.points
        ]

        canvas.setStrokeColor(_TEAL)
        canvas.setLineWidth(1.5)
        for index in range(1, len(self.points)):
            canvas.line(x_values[index - 1], y_values[index - 1], x_values[index], y_values[index])
        for index, point in enumerate(self.points):
            canvas.setFillColor(_SAMPLE if point["synthetic"] else _TEAL)
            canvas.circle(x_values[index], y_values[index], 3, stroke=0, fill=1)

        label_indices = {0, len(self.points) - 1}
        if len(self.points) > 3:
            label_indices.add(len(self.points) // 2)
        canvas.setFillColor(_NAVY)
        canvas.setFont("Helvetica", 7)
        for index in sorted(label_indices):
            timestamp = self.points[index]["date"]
            label = timestamp.strftime("%m/%d/%y") if timestamp else f"Session {index + 1}"
            canvas.drawCentredString(x_values[index], bottom - 13, label)

        legend_y = 8
        canvas.setFillColor(_TEAL)
        canvas.circle(left, legend_y + 2, 3, stroke=0, fill=1)
        canvas.setFillColor(_NAVY)
        canvas.drawString(left + 7, legend_y, "Saved session")
        if any(point["synthetic"] for point in self.points):
            legend_x = left + 85
            canvas.setFillColor(_SAMPLE)
            canvas.circle(legend_x, legend_y + 2, 3, stroke=0, fill=1)
            canvas.setFillColor(_NAVY)
            canvas.drawString(legend_x + 7, legend_y, "DEMO / SAMPLE")


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


def _draw_page_header_footer(canvas, doc):
    canvas.saveState()
    page_width, page_height = letter
    canvas.setFillColor(_TEAL)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(42, page_height - 25, "NEURONEXUS")
    canvas.setStrokeColor(_GRID)
    canvas.setLineWidth(0.7)
    canvas.line(42, page_height - 32, page_width - 42, page_height - 32)
    canvas.line(42, 30, page_width - 42, 30)
    canvas.setFillColor(_NAVY)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(42, 18, "Research screening summary · Not a diagnosis")
    canvas.drawRightString(page_width - 42, 18, f"Page {doc.page}")
    canvas.restoreState()


def _recorded_feature_rows(report):
    return [
        row for row in key_measurement_rows(report)
        if row["Recorded value"] != NOT_COMPLETED
    ]


def make_pdf(
    name,
    report,
    model_status=None,
    *,
    session_id=None,
    session_created_at=None,
    history=None,
    wellness_plan=None,
):
    if report is None:
        report = {}
    if not isinstance(report, Mapping):
        raise ValueError("A screening report must be an object.")
    if session_id is None:
        session_id = report.get("session_id")
    history_records = _history_records(history, session_id=session_id)
    out = BytesIO()
    doc = SimpleDocTemplate(
        out,
        pagesize=letter,
        rightMargin=42,
        leftMargin=42,
        topMargin=52,
        bottomMargin=42,
        pageCompression=0,
    )
    styles = getSampleStyleSheet()
    styles["Title"].fontName = "Helvetica-Bold"
    styles["Title"].textColor = _NAVY
    styles["Title"].fontSize = 21
    styles["Title"].leading = 26
    styles["Heading2"].fontName = "Helvetica-Bold"
    styles["Heading2"].textColor = _TEAL
    styles["Heading2"].spaceBefore = 12
    styles["Heading2"].spaceAfter = 6
    styles["Heading3"].textColor = _NAVY
    styles["BodyText"].fontName = "Helvetica"
    styles["BodyText"].textColor = _NAVY
    styles["BodyText"].leading = 13
    consent_flags = report.get("consent_flags") or {}
    participant_identifier = None
    if consent_flags.get("report_participant_identifier") is True:
        participant_identifier = report.get("participant_identifier") or name
    completed_modules = completed_module_explanations(report)
    metadata_rows = [
        ["Session ID", _display_value(session_id) if session_id is not None else "Not assigned — report not saved"],
        ["Assessment date and time", _format_datetime(session_created_at or report.get("created_utc"))],
        ["Participant identifier", participant_identifier or "Not included (privacy)"],
        ["Screening type", _display_value(report.get("workflow_mode"))],
        [
            "Session status",
            "Assessment results recorded"
            if completed_modules
            else "Incomplete — no completed assessments recorded",
        ],
    ]
    story = [
        _paragraph("NeuroNexus", styles["Title"]),
        _paragraph("Screening session report", styles["Heading2"]),
        _table([["Report details", "Recorded value"], *metadata_rows], [180, 348]),
        Spacer(1, 12),
        _how_to_read_box(report, styles, doc.width),
        Spacer(1, 14),
        _paragraph(
            "This research screening summary is not a diagnosis and cannot confirm "
            "or rule out a neurological condition. Clinical interpretation requires "
            "a qualified healthcare professional.",
            styles["BodyText"],
        ),
        Spacer(1, 14),
    ]

    if is_synthetic_session(report):
        story[2:2] = [
            _paragraph("DEMO / SAMPLE — Not a patient record", styles["Heading2"]),
            _paragraph(
                report.get("demo_notice")
                or "Synthetic sample data; not a patient record.",
                styles["BodyText"],
            ),
            Spacer(1, 8),
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
    memory = cognitive.get("memory_check") or {}
    memory_completed = bool(memory.get("completed"))
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
        [
            "Memory check",
            "Completed" if memory_completed else NOT_COMPLETED,
            "Prototype timed word recall; not diagnostic" if memory_completed else NOT_COMPLETED,
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

    voice = report.get("voice") or {}
    story.append(_paragraph("Audio assessment summary", styles["Heading2"]))
    if voice:
        audio_summary = [
            ["Measurement", "Recorded value"],
            ["Recording duration", _display_value(voice.get("duration_seconds"))],
            ["Median pitch", _display_value(voice.get("pitch_median_hz"))],
            ["Signal quality", _display_value((voice.get("quality") or {}).get("label"))],
            ["Signal quality score", _display_value((voice.get("quality") or {}).get("score"))],
        ]
        story.extend([
            _table(audio_summary, [200, 328]),
            Spacer(1, 10),
            _paragraph("Audio processing information", styles["Heading3"]),
        ])
        processing_fields = (
            ("Source sample rate", "source_sample_rate", "Hz"),
            ("Source channels", "source_channels", "channels"),
            ("Source duration", "source_duration_seconds", "seconds"),
            ("Analysis sample rate", "sample_rate", "Hz"),
            ("Conversion method", "conversion_method", ""),
        )
        processing_rows = [["Processing detail", "Recorded value"]]
        for label, key, unit in processing_fields:
            value = voice.get(key)
            if value is not None:
                processing_rows.append([
                    label,
                    f"{_display_value(value)} {unit}".strip(),
                ])
        if len(processing_rows) > 1:
            story.append(_table(processing_rows, [200, 328]))
        else:
            story.append(_paragraph(
                "Processing metadata is not available in the saved report.",
                styles["BodyText"],
            ))
    else:
        story.append(_paragraph("No audio assessment was recorded.", styles["BodyText"]))
    story.append(Spacer(1, 12))

    model_outputs = report.get("model_outputs")
    story.extend([
        _paragraph("Participant-specific model output", styles["Heading2"]),
        _paragraph(
            _display_value(model_outputs)
            if model_outputs
            else (
                "No participant-specific model output is recorded for this session. "
                "Research model availability and cohort metrics are not individual results."
            ),
            styles["BodyText"],
        ),
        Spacer(1, 8),
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
        _paragraph(
            "Features actually extracted / Key measurement breakdown",
            styles["Heading2"],
        ),
        _paragraph(
            "Only measurements recorded in this report are listed. Missing values "
            "are not estimated; no validated healthy reference ranges are available.",
            styles["BodyText"],
        ),
        Spacer(1, 6),
    ])
    measurements = _recorded_feature_rows(report)
    if measurements:
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
        story.append(_table(measurement_table, [72, 120, 100, 112, 124], font_size=7))
    else:
        story.append(_paragraph(
            "No feature measurements were recorded for this session.",
            styles["BodyText"],
        ))
    story.append(Spacer(1, 14))

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

    story.append(_paragraph("Memory assessment", styles["Heading2"]))
    if memory_completed:
        memory_rows = [["Memory task result", "Recorded value"]]
        for label, key, unit in (
            ("Items recalled", "items_recalled", "items"),
            ("Items presented", "items_presented", "words"),
            ("Recall percentage", "recall_percentage", "%"),
            ("Recall entry time", "recall_latency_seconds", "seconds"),
            ("Completion time", "completion_time_seconds", "seconds"),
        ):
            value = memory.get(key)
            if value is not None:
                memory_rows.append([label, f"{_display_value(value)} {unit}".strip()])
        story.extend([
            _table(memory_rows, [200, 328]),
            _paragraph(
                "This prototype task is descriptive, not a validated clinical assessment.",
                styles["BodyText"],
            ),
        ])
    else:
        story.append(_paragraph(
            "No completed memory assessment was recorded for this session.",
            styles["BodyText"],
        ))
    story.append(Spacer(1, 10))

    story.append(_paragraph("Brain-health session history", styles["Heading2"]))
    story.append(_paragraph(
        "History contains saved sessions only. Differences are descriptive and do not "
        "indicate disease progression or clinical significance.",
        styles["BodyText"],
    ))
    if history_records:
        history_table = [[
            "Assessment date",
            "Session ID",
            "Type",
            "Completed assessments",
            "Data label",
        ]]
        for record in history_records:
            history_table.append([
                _format_datetime(record["timestamp"]),
                _display_value(record["id"]),
                _display_value(record["type"]),
                ", ".join(record["assessments"]) or NOT_COMPLETED,
                "DEMO / SAMPLE" if record["synthetic"] else "Saved session",
            ])
        story.extend([
            _table(history_table, [92, 48, 70, 207, 111], font_size=7),
            Spacer(1, 8),
        ])
        chart_count = 0
        for label in TIMELINE_METRICS:
            points = [
                {
                    "date": record["date"],
                    "value": record["measurements"][label],
                    "synthetic": record["synthetic"],
                }
                for record in history_records
                if record["date"] is not None
                and label in record["measurements"]
            ]
            if len(points) < 2:
                continue
            points.sort(key=lambda point: _datetime_sort_key(point["date"]))
            unit = label.rsplit("(", 1)[-1].rstrip(")")
            story.append(_HistoryLineChart(f"{label} · descriptive history", points))
            story.append(_paragraph(
                f"Recorded values in {unit}; no clinical direction or significance is inferred.",
                styles["BodyText"],
            ))
            story.append(Spacer(1, 8))
            chart_count += 1
            if chart_count >= 2:
                break
        if not chart_count:
            story.append(_paragraph(
                "A history chart is not shown because fewer than two dated values "
                "were available for the same measurement.",
                styles["BodyText"],
            ))
    else:
        story.append(_paragraph(
            "No saved assessment history was available when this report was generated.",
            styles["BodyText"],
        ))
    story.append(Spacer(1, 10))

    story.append(_paragraph("Wellness recommendations", styles["Heading2"]))
    if wellness_plan:
        goals = wellness_plan.get("goals") or []
        if goals:
            story.append(_paragraph(
                "Selected goals: " + ", ".join(map(str, goals)),
                styles["BodyText"],
            ))
        recommendations = wellness_plan.get("recommendations") or []
        if recommendations:
            for recommendation in recommendations:
                story.append(_paragraph(f"- {recommendation}", styles["BodyText"]))
        else:
            story.append(_paragraph(
                "No tailored suggestions were generated from the saved preferences.",
                styles["BodyText"],
            ))
        story.append(_paragraph(
            "These are optional, general educational ideas based on goals and "
            "preferences the user chose to save. They are not medical advice.",
            styles["BodyText"],
        ))
    else:
        story.append(_paragraph(
            "No saved wellness plan was available; no individualized wellness "
            "recommendations are included.",
            styles["BodyText"],
        ))
    story.append(Spacer(1, 10))

    story.append(_paragraph("Limitations", styles["Heading2"]))
    story.extend([
        _paragraph(
            report.get("purpose")
            or "Measurements are exploratory and descriptive, not a diagnosis or disease-risk estimate.",
            styles["BodyText"],
        ),
        _paragraph(
            "Results may vary with device, recording conditions, task familiarity, "
            "language, fatigue, and other context. Research model metrics, when shown, "
            "describe cohort evaluation and are not personal predictions.",
            styles["BodyText"],
        ),
        Spacer(1, 8),
    ])

    story.append(_paragraph(
        "Clinical review guidance / Clinician Discussion Guide",
        styles["Heading2"],
    ))
    movement_recorded = bool(report.get("resting_motion") or report.get("gait"))
    cognition_recorded = bool(
        clock.get("completed")
        or reaction_completed
        or memory_completed
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
            "For persistent concerns, discuss symptoms and changes with a qualified "
            "healthcare professional. Seek immediate medical care for sudden new "
            "neurological symptoms.",
            styles["BodyText"],
        ),
    ])

    story.extend([
        _paragraph("Privacy statement", styles["Heading2"]),
        _paragraph(
            "This export includes the derived measurements and optional saved "
            "history or wellness recommendations shown above; it does not include "
            "raw audio or sensor uploads. Participant identity is omitted unless "
            "separately permitted. Review this document before sharing it.",
            styles["BodyText"],
        ),
    ])

    doc.build(
        story,
        onFirstPage=_draw_page_header_footer,
        onLaterPages=_draw_page_header_footer,
    )
    return out.getvalue()
