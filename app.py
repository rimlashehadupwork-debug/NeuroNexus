from datetime import date, datetime, timezone
import base64, json, logging, random, re, time
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from modules.config import (
    APP_NAME,
    APP_VERSION,
    CONSENT_INFORMATION,
    CONSENT_TEXT,
    DIAGNOSTIC_DISCLAIMER,
    PRIVACY_NOTICE,
    RESPONSIBLE_AI_NOTICE,
    STORAGE_ARCHITECTURE_NOTE,
    SUPPORTED_MODES,
)
from modules.audio_analysis import (
    MAX_AUDIO_UPLOAD_BYTES,
    analyze_audio_bytes,
    audio_visualization_data,
)
from modules.motion_analysis import analyze_motion, analyze_gait, read_sensor_csv
from modules.session import (
    assessment_completion,
    assessment_draft_payload,
    clear_assessment_inputs,
    has_assessment_progress,
    initialize_session,
    reset_session,
    restore_assessment_draft,
)
from modules.reporting import (
    build_report,
    completed_module_explanations,
    humanize_feature_code,
    key_measurement_rows,
    report_json,
)
from modules.accounts import (
    authenticate,
    caregiver_sessions,
    delete_wellness_plan,
    delete_draft,
    delete_user_sessions,
    init_db,
    is_duplicate_email_error,
    load_draft,
    load_wellness_plan,
    register,
    save_draft,
    save_session,
    save_wellness_plan,
    storage_backend,
    update_consent_flag,
    update_profile,
    user_sessions,
    admin_overview,
)
from modules.pdf_report import make_pdf
from modules.cognitive_tasks import reaction_summary
from modules.memory_check import render_memory_check
from modules.longitudinal import (
    TIMELINE_METRICS,
    is_synthetic_session,
    session_metric_values,
    session_report,
)
from modules.wellness import (
    LIFESTYLE_OPTIONS,
    WELLNESS_GOALS,
    build_wellness_plan,
    daily_tip,
    new_saved_wellness_state,
    normalize_wellness_state,
    wellness_progress_rows,
)
from modules.demo_mode import (
    DEMO_MODE_NOTICE,
    enter_demo_mode,
    is_demo_mode,
    reset_demo_mode,
    save_demo_report,
    sessions_for_user,
)
from modules.clock_drawing import extract_clock_drawing_features, assess_darwin_feature_compatibility
from modules.model_registry import get_registry          # ← NEW
from modules.alzheimers_model import get_alzheimers_status, load_darwin_sample_profile  # ← NEW
from modules.parkinson_model import train_parkinson_model
from modules.praat_features import (
    PRAAT_FEATURE_EXPLANATIONS,
    run_symptom_pattern_analysis,
)

logger = logging.getLogger(__name__)

LUCIDE_ICONS = {
    'health': (
        '<svg class="nn-lucide" viewBox="0 0 24 24" aria-hidden="true" focusable="false">'
        '<path d="M22 12h-3l-3 9L10 3l-3 9H2"/></svg>'
    ),
    'voice': (
        '<svg class="nn-lucide" viewBox="0 0 24 24" aria-hidden="true" focusable="false">'
        '<rect x="9" y="2" width="6" height="12" rx="3"/>'
        '<path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3m-4 0h8"/></svg>'
    ),
    'cognition': (
        '<svg class="nn-lucide" viewBox="0 0 24 24" aria-hidden="true" focusable="false">'
        '<path d="M12 18V5m3 8a4 4 0 0 1-3-4 4 4 0 0 1-3 4"/>'
        '<path d="M17.6 6.5A3 3 0 1 0 12 5a3 3 0 1 0-5.6 1.5M18 18a4 4 0 0 0 2-7.5 4 4 0 0 0-2.5-5.4M6 18a4 4 0 0 1-2-7.5 4 4 0 0 1 2.5-5.4M14.4 15a4 4 0 0 0 3.2 2m-7.2-2a4 4 0 0 1-3.2 2"/></svg>'
    ),
    'movement': (
        '<svg class="nn-lucide" viewBox="0 0 24 24" aria-hidden="true" focusable="false">'
        '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>'
    ),
}

st.set_page_config(page_title='NeuroNexus | Digital Biomarker Research', page_icon='🧠', layout='wide', initial_sidebar_state='expanded')
initialize_session(st.session_state)
for k,v in {'user':None,'reaction_times':[],'reaction_started':None,'reaction_ready':False,'clock_result':None}.items():
    if k not in st.session_state: st.session_state[k]=v
if st.session_state.user and not is_demo_mode(st.session_state.user):
    init_db()
if (
    st.session_state.get("user")
    and not is_demo_mode(st.session_state.user)
    and st.session_state.get("consent_granted", False)
    and has_assessment_progress(st.session_state)
    and not st.session_state.get("draft_finalized", False)
):
    save_draft(
        st.session_state.user["id"],
        assessment_draft_payload(st.session_state),
    )

def navigate_workspace(page, directory_role=None):
    st.session_state['workspace_page']=page
    if directory_role is not None:
        st.session_state['directory_role']=directory_role


def decline_assessment_consent():
    st.session_state["assessment_consent_checked"] = False
    navigate_workspace("My dashboard")


def mark_assessment_changed():
    st.session_state.draft_finalized=False
    st.session_state.last_saved_report_session_id=None
    st.session_state.last_saved_report_created_at=None
    st.session_state.report_saved_notice=None


def saved_wellness_report_context(user):
    if is_demo_mode(user):
        payload=st.session_state.get('demo_wellness_plan')
    else:
        payload=load_wellness_plan(user['id'])
    if not payload:
        return None
    saved_plan=normalize_wellness_state(payload)
    plan=build_wellness_plan(
        [],
        saved_plan['lifestyle'],
        saved_plan['goals'],
    )
    return {
        'goals':[WELLNESS_GOALS[goal] for goal in saved_plan['goals']],
        'recommendations':plan['habit_ideas'],
        'saved_at':saved_plan['saved_at'],
    }


def render_brain_health_timeline(
    sessions,
    key_prefix,
    participant_name=None,
    show_start_button=False,
    wellness_plan=None,
):
    st.subheader('Brain health timeline')
    st.caption(
        'Repeated assessments may help visualize measurements over time. Changes '
        'are descriptive only and do not indicate disease progression or clinical '
        'significance. Device, task, and context differences can affect results.'
    )
    if participant_name:
        st.caption(f'Participant: {participant_name}')
    if not sessions:
        st.info(
            'Your assessment history will appear here as you complete sessions. '
            'Repeated assessments may help visualize changes over time, but this '
            'timeline is not a diagnostic tool.'
        )
        if show_start_button:
            st.button(
                'Start an assessment',
                type='primary',
                on_click=navigate_workspace,
                args=('Screening assessments',),
                key=f'{key_prefix}_timeline_start_assessment',
            )
        return

    entries = []
    try:
        for index, session in enumerate(sessions):
            report = session_report(session)
            timestamp = session.get('created_at') or report.get('created_utc')
            session_date = pd.to_datetime(timestamp, utc=True, errors='coerce')
            if pd.isna(session_date):
                session_date = None
            modules = completed_module_explanations(report)
            session_id = session.get('id')
            entries.append({
                'id': session_id,
                'key': f'{session_id if session_id is not None else "session"}-{index}',
                'date': session_date,
                'created_at': timestamp,
                'date_label': (
                    session_date.strftime('%b %d, %Y · %I:%M %p UTC')
                    if session_date is not None
                    else 'Date unavailable'
                ),
                'type': str(report.get('workflow_mode') or 'Not recorded'),
                'report': report,
                'modules': modules,
                'measurements': session_metric_values(report),
                'synthetic': is_synthetic_session(report),
                'summary': session.get('summary') or {},
            })
    except ValueError as exc:
        st.error(f'A saved session report could not be read: {exc}')
        return

    entries.sort(
        key=lambda entry: (
            entry['date']
            if entry['date'] is not None
            else pd.Timestamp.min.tz_localize('UTC')
        ),
        reverse=True,
    )
    synthetic_count = sum(entry['synthetic'] for entry in entries)
    if synthetic_count:
        st.warning(
            f'{synthetic_count} session(s) are synthetic examples, not patient history.',
            title='DEMO / SAMPLE data',
        )

    dated_entries = [entry for entry in entries if entry['date'] is not None]
    unique_domains = {
        module
        for entry in entries
        for module in entry['modules']
    }
    overview = st.columns(3)
    overview[0].metric('Sessions shown', len(entries))
    overview[1].metric('Assessment domains recorded', len(unique_domains))
    overview[2].metric(
        'Most recent session',
        dated_entries[0]['date'].strftime('%b %d, %Y')
        if dated_entries
        else 'Date unavailable',
    )

    session_types = sorted({entry['type'] for entry in entries})
    assessments = sorted({
        module
        for entry in entries
        for module in entry['modules']
    })
    filter_cols = st.columns(2)
    selected_type = filter_cols[0].selectbox(
        'Session type',
        ['All session types', *session_types],
        key=f'{key_prefix}_timeline_type',
    )
    selected_assessment = filter_cols[1].selectbox(
        'Completed assessment',
        ['All assessments', *assessments],
        key=f'{key_prefix}_timeline_assessment',
    )
    filtered_entries = [
        entry for entry in entries
        if (selected_type == 'All session types' or entry['type'] == selected_type)
        and (
            selected_assessment == 'All assessments'
            or selected_assessment in entry['modules']
        )
    ]

    st.markdown('#### Session timeline')
    if not filtered_entries:
        st.info('No sessions match the selected filters.')
        return
    for entry in filtered_entries:
        report = entry['report']
        cognitive = report.get('cognitive_tasks') or {}
        memory = cognitive.get('memory_check') or {}
        voice = report.get('voice') or {}
        voice_pattern = report.get('voice_pattern_features') or {}
        with st.container(border=True):
            title_col, button_col = st.columns([4, 1], vertical_alignment='center')
            title_col.markdown(f"**{entry['date_label']}**")
            type_label = f"Session type: {entry['type']}"
            if entry['synthetic']:
                type_label += ' · DEMO / SAMPLE'
            title_col.caption(type_label)
            title_col.write(
                ', '.join(entry['modules']) if entry['modules']
                else 'No completed assessments recorded.'
            )
            card_details = []
            if voice or voice_pattern:
                duration = voice.get('duration_seconds')
                pitch_variation = voice_pattern.get('std_pitch')
                audio_parts = ['Audio recorded']
                if isinstance(duration, (int, float)):
                    audio_parts.append(f'{duration:.1f} s')
                if isinstance(pitch_variation, (int, float)):
                    audio_parts.append(f'pitch variation {pitch_variation:.2g} Hz')
                card_details.append(' · '.join(audio_parts))
            if memory.get('completed'):
                recalled = memory.get('items_recalled')
                presented = memory.get('items_presented')
                if recalled is not None and presented is not None:
                    card_details.append(f'Memory check: {recalled} of {presented} words recalled')
                else:
                    card_details.append('Memory check completed; score unavailable')
            if card_details:
                title_col.caption(' · '.join(card_details))
            if button_col.button(
                'View details',
                key=f'{key_prefix}_timeline_view_{entry["key"]}',
            ):
                st.session_state[f'{key_prefix}_timeline_selected'] = entry['key']

    metric_options = [
        label for label in TIMELINE_METRICS
        if any(label in entry['measurements'] for entry in filtered_entries)
    ]
    if not metric_options:
        st.info('No comparable dated numeric measurements are available in these sessions.')
    else:
        st.markdown('#### Recorded measurements')
        chart_cols = st.columns(2)
        selected_metric = chart_cols[0].selectbox(
            'Measurement',
            metric_options,
            key=f'{key_prefix}_timeline_metric',
        )
        time_window = chart_cols[1].selectbox(
            'Time window',
            ['30 days', '60 days', '90 days', 'All saved sessions'],
            index=1,
            key=f'{key_prefix}_timeline_window',
        )
        chart_entries = [
            entry for entry in filtered_entries
            if entry['date'] is not None and selected_metric in entry['measurements']
        ]
        if time_window != 'All saved sessions':
            days = int(time_window.split()[0])
            cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days)
            chart_entries = [
                entry for entry in chart_entries if entry['date'] >= cutoff
            ]
        if not chart_entries:
            st.info('No dated sessions with this measurement fall in the selected time window.')
        else:
            chart_data = pd.DataFrame([
                {
                    'Session date': entry['date'],
                    'Recorded value': entry['measurements'][selected_metric],
                }
                for entry in sorted(chart_entries, key=lambda item: item['date'])
            ])
            st.line_chart(
                chart_data,
                x='Session date',
                y='Recorded value',
                alt=f'{selected_metric} recorded across saved sessions; descriptive only',
            )

            first_value = float(chart_data.iloc[0]['Recorded value'])
            latest_value = float(chart_data.iloc[-1]['Recorded value'])
            if len(chart_data) > 1:
                unit = selected_metric.rsplit('(', 1)[-1].rstrip(')')
                st.metric(
                    'First-to-latest recorded difference',
                    f'{latest_value - first_value:+.2f} {unit}',
                    delta=None,
                    help=(
                        f'Descriptive difference across {len(chart_data)} recorded sessions. '
                        'It is not a clinical change score or evidence of progression.'
                    ),
                )
                st.caption(
                    'This difference is descriptive, not a clinical change score. '
                    'No direction is interpreted as better or worse.'
                )

    selected_key = st.session_state.get(f'{key_prefix}_timeline_selected')
    selected_entry = next(
        (entry for entry in filtered_entries if entry['key'] == selected_key),
        filtered_entries[0],
    )
    st.markdown('#### Selected session details')
    with st.container(border=True):
        st.caption(selected_entry['date_label'])
        st.write(f"**Assessment type:** {selected_entry['type']}")
        report = selected_entry['report']
        has_report = bool(report)
        st.write(
            '**Processing status:** '
            + (
                'Finalized report saved; processing-stage details were not recorded.'
                if has_report
                else 'No saved report summary is available.'
            )
        )
        if selected_entry['synthetic']:
            st.warning('DEMO / SAMPLE — synthetic example, not a patient record.')

        modules = selected_entry['modules']
        if modules:
            st.markdown('**Completed assessments**')
            for name, description in modules.items():
                st.markdown(f'- **{name}:** {description}')
        else:
            st.info('No completed assessments are recorded for this session.')

        voice = report.get('voice') or {}
        voice_pattern = report.get('voice_pattern_features') or {}
        if voice or voice_pattern:
            st.markdown('**Audio assessment summary**')
            audio_summary = {
                'Recording duration (seconds)': voice.get('duration_seconds'),
                'Median pitch (Hz)': voice.get('pitch_median_hz'),
                'Pitch variation (Hz)': voice_pattern.get('std_pitch'),
                'Signal quality': (voice.get('quality') or {}).get('label'),
            }
            st.table(
                {
                    'Measurement': list(audio_summary),
                    'Recorded value': [
                        f'{value:.4g}' if isinstance(value, float)
                        else str(value) if value is not None
                        else 'Not recorded'
                        for value in audio_summary.values()
                    ],
                },
                border='horizontal',
                width='stretch',
                alt='Recorded audio assessment measurements for this session',
            )
        else:
            st.markdown('**Audio assessment summary**')
            st.write('Not recorded in this session.')

        cognitive = report.get('cognitive_tasks') or {}
        memory = cognitive.get('memory_check') or {}
        st.markdown('**Memory task summary**')
        if memory.get('completed'):
            recalled = memory.get('items_recalled')
            presented = memory.get('items_presented')
            st.write(
                f'{recalled} of {presented} words recalled'
                if recalled is not None and presented is not None
                else 'Completed; score not recorded.'
            )
            if memory.get('recall_percentage') is not None:
                st.caption(f"Recorded recall: {memory['recall_percentage']}%")
        else:
            st.write('Not completed or not recorded.')
        st.caption('This prototype task is descriptive and is not a validated clinical test.')

        st.markdown('**Model output**')
        model_outputs = report.get('model_outputs')
        if model_outputs:
            st.json(model_outputs)
        else:
            st.write(
                'No participant-specific model output is recorded. Research model '
                'availability and cohort metrics are not individual results.'
            )
            clock = cognitive.get('clock_drawing') or {}
            compatibility = clock.get('darwin_model_compatibility') or {}
            if compatibility.get('message'):
                st.caption(f"Research-model compatibility: {compatibility['message']}")
        if report.get('darwin_sample_profile'):
            st.info(
                'A research sample profile is attached; it is a cohort example, '
                'not a participant measurement or model output.'
            )

        recorded_rows = [
            row for row in key_measurement_rows(report)
            if row['Recorded value'] != 'Not completed'
        ]
        st.markdown('**Features measured**')
        if recorded_rows:
            st.dataframe(
                pd.DataFrame(recorded_rows),
                hide_index=True,
                alt='Recorded session features and measurements',
            )
        else:
            st.write('No numeric or task measurements were saved for this session.')
        st.caption('Measurements absent from the saved report are not estimated.')

        chronological = sorted(
            (entry for entry in entries if entry['date'] is not None),
            key=lambda item: item['date'],
        )
        prior_entries = [
            entry for entry in chronological
            if entry['date'] < selected_entry['date']
        ] if selected_entry['date'] is not None else []
        prior = next(
            (
                entry for entry in reversed(prior_entries)
                if set(entry['measurements']) & set(selected_entry['measurements'])
            ),
            None,
        )
        st.markdown('**Session-to-session differences**')
        if prior:
            st.caption(
                f"Compared with the nearest earlier session with shared measurements "
                f"({prior['date_label']}). Differences are descriptive only."
            )
            shared = [
                label for label in TIMELINE_METRICS
                if label in prior['measurements']
                and label in selected_entry['measurements']
            ]
            comparison_cols = st.columns(min(len(shared), 3))
            for index, label in enumerate(shared):
                difference = (
                    selected_entry['measurements'][label]
                    - prior['measurements'][label]
                )
                unit = label.rsplit('(', 1)[-1].rstrip(')')
                comparison_cols[index % len(comparison_cols)].metric(
                    label,
                    f'{difference:+.2f} {unit}',
                    delta=None,
                    help='Recorded difference only; no clinical significance is inferred.',
                )
        else:
            st.write('No earlier session has a comparable recorded measurement.')

        st.markdown('**Limitations**')
        st.write(
            report.get('purpose')
            or 'Measurements are exploratory and descriptive; this timeline is not diagnostic.'
        )
        st.caption(
            'Differences may reflect task, device, recording, or context variation. '
            'No diagnosis, disease progression, or clinical significance is inferred.'
        )

        if has_report:
            json_data = (
                selected_entry['summary']
                if isinstance(selected_entry['summary'], str)
                else json.dumps(report, indent=2, allow_nan=False)
            )
            report_id = selected_entry['id'] or selected_entry['key']
            export_cols = st.columns(2)
            export_cols[0].download_button(
                'Download session JSON',
                json_data,
                file_name=f'neuronexus_session_{report_id}.json',
                mime='application/json',
                key=f'{key_prefix}_timeline_json_{selected_entry["key"]}',
            )
            export_cols[1].download_button(
                'Download session PDF',
                make_pdf(
                    participant_name or 'Participant',
                    report,
                    get_registry(),
                    session_id=selected_entry['id'],
                    session_created_at=selected_entry['created_at'],
                    history=sessions,
                    wellness_plan=wellness_plan,
                ),
                file_name=f'neuronexus_session_{report_id}.pdf',
                mime='application/pdf',
                key=f'{key_prefix}_timeline_pdf_{selected_entry["key"]}',
            )


def render_wellness_coach(sessions, key_prefix, user):
    st.title('Brain wellness companion')
    st.caption('Supportive, general information and tools for routines you choose.')
    st.info(
        'This companion offers general educational guidance. It does not diagnose, '
        'prescribe, prevent or treat disease, or replace advice from a qualified clinician.',
        title='Wellness, not medical care',
    )
    st.caption(
        'Goals and self-reported preferences are optional. They are saved to your '
        'account only when you choose Save; they are not added to assessment reports. '
        'Demo Mode keeps this plan in the current browser session only.'
    )

    completed_domains = set()
    for session in sessions:
        completed_domains.update(
            completed_module_explanations(session_report(session))
        )

    demo_mode = is_demo_mode(user)
    owner_key = 'demo' if demo_mode else str(user['id'])
    generation_key = f'{key_prefix}_wellness_generation_{owner_key}'
    generation = st.session_state.get(generation_key, 0)
    widget_namespace = (
        f'{key_prefix}_wellness_{owner_key}_{generation}'
    )
    if demo_mode:
        saved_payload = st.session_state.get('demo_wellness_plan')
    else:
        with st.spinner('Loading your saved wellness plan...'):
            saved_payload = load_wellness_plan(user['id'])
    wellness_state = normalize_wellness_state(saved_payload)
    today = date.today()
    today_key = today.isoformat()

    with st.container(border=True):
        st.subheader("Today's wellness tip")
        st.write(daily_tip(today))
        st.caption(
            'Tips and activities are broad suggestions. They are not a promise of '
            'health outcomes; choose only what feels suitable for you.'
        )

    st.subheader('Choose your goals')
    goal_options = list(WELLNESS_GOALS)
    goal_key = f'{widget_namespace}_goals'
    selected_goals = st.multiselect(
        'Which areas would you like to focus on?',
        goal_options,
        default=wellness_state['goals'],
        format_func=lambda goal: WELLNESS_GOALS[goal],
        key=goal_key,
    )
    st.caption(
        'You can choose none, one, or several. With no goals selected, the general '
        'activity list remains available.'
    )

    st.subheader('Optional self-reported preferences')
    lifestyle_labels = {
        'sleep_quality': 'How has your sleep quality been?',
        'physical_activity': 'How often do you do movement that feels manageable?',
        'social_engagement': 'How connected have you felt to other people?',
        'hydration': 'How often do you remember to drink fluids?',
        'mental_engagement': 'How often do you enjoy mentally engaging activities?',
    }
    lifestyle = {}
    left, right = st.columns(2)
    for index, (field, options) in enumerate(LIFESTYLE_OPTIONS.items()):
        column = left if index % 2 == 0 else right
        with column:
            saved_value = wellness_state['lifestyle'].get(field, options[0])
            if saved_value not in options:
                saved_value = options[0]
            lifestyle[field] = st.selectbox(
                lifestyle_labels[field],
                options,
                index=options.index(saved_value),
                key=f'{widget_namespace}_{field}',
            )

    wellness_plan = build_wellness_plan(
        completed_domains,
        lifestyle,
        selected_goals,
    )
    st.subheader('Suggested activities')
    st.caption(
        'Suggestions are matched only to the goals and preferences you selected, '
        'not to screening scores or model outputs.'
    )
    activities = wellness_plan['suggested_activities']
    recorded_today = set(wellness_state['completed_by_date'].get(today_key, []))
    selected_activity_ids = set()
    activity_columns = st.columns(2)
    for index, activity in enumerate(activities):
        with activity_columns[index % len(activity_columns)]:
            with st.container(border=True):
                activity_key = f'{widget_namespace}_activity_{activity["id"]}'
                checked = st.checkbox(
                    activity['title'],
                    value=activity['id'] in recorded_today,
                    key=activity_key,
                )
                st.caption(activity['description'])
                if checked:
                    selected_activity_ids.add(activity['id'])

    if st.button(
        'Save wellness plan and today’s activity',
        type='primary',
        key=f'{widget_namespace}_save_plan',
    ):
        active_activity_ids = {activity['id'] for activity in activities}
        completed_by_date = {
            day: list(activity_ids)
            for day, activity_ids in wellness_state['completed_by_date'].items()
        }
        completed_by_date[today_key] = sorted(
            (
                recorded_today - active_activity_ids
            ) | selected_activity_ids
        )
        wellness_state = new_saved_wellness_state(
            selected_goals,
            lifestyle,
            completed_by_date,
        )
        if demo_mode:
            st.session_state['demo_wellness_plan'] = wellness_state
        else:
            save_wellness_plan(user['id'], wellness_state)
        st.success('Your wellness plan and today’s activity were saved.')

    if wellness_state['saved_at']:
        st.caption(f"Saved plan · {wellness_state['saved_at'][:10]}")
        if st.button(
            'Delete saved wellness plan',
            key=f'{widget_namespace}_delete_plan',
        ):
            if demo_mode:
                st.session_state.pop('demo_wellness_plan', None)
            else:
                delete_wellness_plan(user['id'])
            st.session_state[generation_key] = generation + 1
            st.rerun()
    else:
        st.caption('Your current selections are not saved yet.')

    st.subheader('Your wellness dashboard')
    saved_today = set(wellness_state['completed_by_date'].get(today_key, []))
    saved_active_count = len(saved_today & {activity['id'] for activity in activities})
    tracked_days = sum(
        1 for row in wellness_progress_rows(
            wellness_state['completed_by_date'],
            today,
            days=7,
        )
        if row['Activities marked complete'] is not None
    )
    metrics = st.columns(3)
    metrics[0].metric('Goals selected', len(wellness_state['goals']))
    metrics[1].metric(
        'Activities logged today',
        f'{saved_active_count} of {len(activities)}',
    )
    metrics[2].metric('Days with a saved check-in · last 7 days', tracked_days)
    st.progress(
        saved_active_count / len(activities) if activities else 0.0,
        text='Share of today’s suggested activities marked complete',
    )
    st.caption(
        'Progress reflects only activities you marked in NeuroNexus. It is not a '
        'measure of health or a performance target.'
    )
    progress_rows = wellness_progress_rows(
        wellness_state['completed_by_date'],
        today,
        days=7,
    )
    progress_data = pd.DataFrame(progress_rows)
    if progress_data['Activities marked complete'].notna().any():
        progress_data['Date'] = pd.to_datetime(progress_data['Date'])
        progress_data['Activities marked complete'] = pd.to_numeric(
            progress_data['Activities marked complete'],
            errors='coerce',
        )
        st.bar_chart(
            progress_data,
            x='Date',
            y='Activities marked complete',
            alt='Activities the user marked complete in NeuroNexus over the last seven days',
        )
    else:
        st.info('Your saved activity progress will appear here after your first check-in.')
    st.caption(
        'Untracked dates are left blank. A date with zero means no activities were '
        'marked complete in this app; it does not mean you did nothing.'
    )

    with st.container(border=True):
        st.subheader('Personalized ideas')
        if wellness_plan['habit_ideas']:
            for idea in wellness_plan['habit_ideas']:
                st.markdown(f'- {idea}')
        else:
            st.caption(
                'Choose any optional preferences above to see related ideas. '
                'General daily suggestions remain available without sharing preferences.'
            )
    with st.container(border=True):
        st.subheader('Professional care guidance')
        st.info(
            'If you have neurological symptoms or changes that persist, worsen, or '
            'affect daily life, consider discussing them with a qualified healthcare '
            'professional. This companion cannot determine whether someone is healthy '
            'or unwell.',
            title='For ongoing concerns',
        )
        st.subheader('Clinician visit checklist')
        st.caption(
            'Saved assessment coverage below is included only to help organize a '
            'conversation. Screening measurements and model results are not used to '
            'assess your wellbeing or determine these suggestions.'
        )
        for item in wellness_plan['clinician_checklist']:
            st.markdown(f'- {item}')

    st.warning(
        'Seek immediate medical care for sudden new symptoms such as facial drooping, '
        'weakness or numbness on one side, sudden trouble speaking, severe sudden '
        'headache, fainting, or sudden confusion. Contact local emergency services.',
        title='Urgent symptoms',
    )


def style_research_figure(figure, height=340):
    figure.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#FFFFFF',
        font={'family':'Atkinson Hyperlegible, sans-serif','color':'#172B3B','size':14},
        margin={'l':24,'r':20,'t':58,'b':28},
        height=height,
        hoverlabel={'bgcolor':'#17364A','font':{'color':'#FFFFFF'}},
    )
    figure.update_xaxes(showgrid=False,linecolor='#DCE5EC',zerolinecolor='#DCE5EC')
    figure.update_yaxes(gridcolor='#E8EEF3',linecolor='#DCE5EC',zerolinecolor='#DCE5EC')
    return figure


def clear_pattern_demo():
    st.session_state["pattern_demo_active"] = False


def clear_clock_canvas():
    st.session_state["clock_canvas_generation"] = (
        st.session_state.get("clock_canvas_generation", 0) + 1
    )
    st.session_state["clock_sample_drawing"] = None
    st.session_state.clock_result = None
    mark_assessment_changed()


def _clock_path_object(strokes):
    points = [point for stroke in strokes for point in stroke]
    left = min(point[0] for point in points)
    top = min(point[1] for point in points)
    right = max(point[0] for point in points)
    bottom = max(point[1] for point in points)
    path = []
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        path.append(["M", stroke[0][0] - left, stroke[0][1] - top])
        path.extend(["L", point[0] - left, point[1] - top] for point in stroke[1:])
    return {
        "type": "path",
        "originX": "left",
        "originY": "top",
        "left": left,
        "top": top,
        "width": max(1, right - left),
        "height": max(1, bottom - top),
        "fill": None,
        "stroke": "#0d192b",
        "strokeWidth": 3,
        "strokeLineCap": "round",
        "strokeLineJoin": "round",
        "strokeMiterLimit": 10,
        "scaleX": 1,
        "scaleY": 1,
        "angle": 0,
        "flipX": False,
        "flipY": False,
        "opacity": 1,
        "visible": True,
        "path": path,
    }


def build_sample_clock_drawing():
    center_x, center_y, radius = 260, 170, 112
    outer_circle = [
        (
            center_x + radius * np.cos(angle),
            center_y + radius * np.sin(angle),
        )
        for angle in np.linspace(-np.pi / 2, 3 * np.pi / 2, 49)
    ]
    stroke_groups = [[outer_circle]]
    for hour in range(12):
        angle = -np.pi / 2 + hour * 2 * np.pi / 12
        inner_radius = 94 if hour % 3 == 0 else 99
        stroke_groups.append([[
            (
                center_x + inner_radius * np.cos(angle),
                center_y + inner_radius * np.sin(angle),
            ),
            (
                center_x + 105 * np.cos(angle),
                center_y + 105 * np.sin(angle),
            ),
        ]])

    segments = {
        "0": "abcdef",
        "1": "bc",
        "2": "abged",
        "3": "abgcd",
        "4": "fgbc",
        "5": "afgcd",
        "6": "afgecd",
        "7": "abc",
        "8": "abcdefg",
        "9": "abfgcd",
    }
    segment_lines = {
        "a": ((0, 0), (10, 0)),
        "b": ((10, 0), (10, 9)),
        "c": ((10, 9), (10, 18)),
        "d": ((0, 18), (10, 18)),
        "e": ((0, 9), (0, 18)),
        "f": ((0, 0), (0, 9)),
        "g": ((0, 9), (10, 9)),
    }
    for hour in range(12):
        angle = -np.pi / 2 + hour * 2 * np.pi / 12
        label = str(hour or 12)
        label_width = len(label) * 10 + (len(label) - 1) * 3
        label_left = center_x + 79 * np.cos(angle) - label_width / 2
        label_top = center_y + 79 * np.sin(angle) - 9
        label_strokes = []
        for digit_index, digit in enumerate(label):
            offset_x = label_left + digit_index * 13
            for segment in segments[digit]:
                start, end = segment_lines[segment]
                label_strokes.append([
                    (offset_x + start[0], label_top + start[1]),
                    (offset_x + end[0], label_top + end[1]),
                ])
        stroke_groups.append(label_strokes)

    def clock_point(hour, hand_length):
        angle = -np.pi / 2 + hour * 2 * np.pi / 12
        return (
            center_x + hand_length * np.cos(angle),
            center_y + hand_length * np.sin(angle),
        )

    stroke_groups.extend([
        [[(center_x, center_y), clock_point(11, 48)]],
        [[(center_x, center_y), clock_point(2, 73)]],
        [[
            (center_x - 4, center_y),
            (center_x + 4, center_y),
            (center_x, center_y - 4),
            (center_x - 4, center_y),
        ]],
    ])
    return {"objects": [_clock_path_object(group) for group in stroke_groups]}


def load_sample_clock_drawing():
    st.session_state["clock_canvas_generation"] = (
        st.session_state.get("clock_canvas_generation", 0) + 1
    )
    st.session_state["clock_sample_drawing"] = build_sample_clock_drawing()
    st.session_state.clock_result = None
    mark_assessment_changed()


def generate_sample_vowel_wav():
    sample_rate = 16000
    time_axis = np.arange(sample_rate * 6, dtype=np.float64) / sample_rate
    fundamental = (
        178
        + 2.2 * np.sin(2 * np.pi * 4.3 * time_axis)
        + 0.6 * np.sin(2 * np.pi * 0.7 * time_axis)
    )
    phase = 2 * np.pi * np.cumsum(fundamental) / sample_rate
    formants = (
        (820 + 22 * np.sin(2 * np.pi * 0.35 * time_axis), 110),
        (1320 + 35 * np.sin(2 * np.pi * 0.22 * time_axis), 150),
        (2700 + 45 * np.sin(2 * np.pi * 0.17 * time_axis), 220),
    )
    sample = np.zeros_like(time_axis)
    for harmonic in range(1, 45):
        harmonic_frequency = harmonic * fundamental
        envelope = sum(
            np.exp(-0.5 * ((harmonic_frequency - center) / bandwidth) ** 2)
            for center, bandwidth in formants
        )
        amplitude = (0.025 + envelope) / harmonic**1.15
        sample += amplitude * np.sin(harmonic * phase)
    sample *= 0.92 + 0.08 * np.sin(2 * np.pi * 3.1 * time_axis)
    sample = (0.75 * sample / np.max(np.abs(sample))).astype(np.float32)

    wav_buffer = BytesIO()
    sf.write(wav_buffer, sample, sample_rate, format='WAV', subtype='PCM_16')
    return wav_buffer.getvalue()


st.markdown('''<style>
:root{--brand:#17364A;--sage:#16858A;--bg:#F4F7FA;--line:#DCE5EC;--ink:#172B3B;--muted:#607381;--cyan:#53B9C4;--violet:#7B72B8}
html,body,[class*="css"]{font-family:Arial,sans-serif;color:var(--ink);font-size:17px;line-height:1.55}
h1,h2,h3,h4,.stButton button,.stDownloadButton button{font-family:Arial,sans-serif}
h1,h2,h3,h4{color:var(--ink);letter-spacing:-.025em}
.stApp{background-color:var(--bg);background-image:radial-gradient(ellipse at 94% 4%,rgba(83,185,196,.08),transparent 28%),radial-gradient(ellipse at 5% 72%,rgba(123,114,184,.035),transparent 30%)}
[data-testid="stMain"]{background:transparent}
.block-container{max-width:1360px;padding-top:1.7rem;padding-bottom:3rem}
[data-testid="stHeadingWithActionElements"]:has(> h1):not(.nn-hero [data-testid="stHeadingWithActionElements"]){margin:12px 0 18px;padding:12px 18px;border-left:5px solid var(--sage);border-radius:0 14px 14px 0;background:linear-gradient(90deg,rgba(255,255,255,.92),rgba(255,255,255,.58));box-shadow:0 4px 16px rgba(30,86,92,.035)}
[data-testid="stHeadingWithActionElements"] h1{margin:0;color:var(--brand);font-size:clamp(1.7rem,2.4vw,2.25rem);line-height:1.25}
[data-testid="stHeadingWithActionElements"] h2,[data-testid="stHeadingWithActionElements"] h3{color:var(--brand)}
[data-testid="stHeadingWithActionElements"]:has(> h2),[data-testid="stHeadingWithActionElements"]:has(> h3){margin:18px 0 10px}
[data-testid="stSidebar"]{background:#EDF3F7;border-right:1px solid var(--line)}
[data-testid="stSidebar"] [data-testid="stImage"] img{filter:drop-shadow(0 5px 12px rgba(30,86,92,.1))}
[data-testid="stSidebar"] [role="radiogroup"] label{min-height:48px;padding:5px 10px;border-radius:10px;transition:background-color .18s ease,color .18s ease}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:#E1EEE9}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:#DCEAE5;color:#1E565C;font-weight:700}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{color:var(--muted)}
[data-testid="stMetric"]{background:rgba(255,255,255,.92);border:1px solid var(--line);padding:18px;border-radius:14px;box-shadow:0 3px 12px rgba(23,54,74,.035)}
[data-testid="stMetricLabel"]{font-size:18px}
[data-testid="stMetricValue"]{color:var(--brand)}
[data-testid="stMetric"] [data-testid="stMetricDelta"]{font-size:15px}
[data-testid="stDataFrame"],[data-testid="stTable"]{overflow:hidden;border:1px solid #C6D8D2;border-top:4px solid var(--brand);border-radius:16px;background:#fff;box-shadow:0 12px 30px rgba(30,86,92,.09);transition:box-shadow .2s ease,border-color .2s ease}
[data-testid="stDataFrame"]:hover,[data-testid="stTable"]:hover{border-color:#A9C6BD;box-shadow:0 16px 36px rgba(30,86,92,.13)}
[data-testid="stDataFrame"] iframe{border-radius:12px}
[data-testid="stDataFrame"] [data-testid="stElementContainer"]{background:#fff}
[data-testid="stDataFrame"] [data-testid="stToolbar"]{background:#F4F8F6;border-bottom:1px solid #E1EBE7;padding:5px 10px}
[data-testid="stDataFrame"] [data-testid="stToolbar"] button{min-width:40px;min-height:40px;border-radius:10px;color:var(--brand)}
[data-testid="stDataFrame"] [data-testid="stToolbar"] button:hover{background:#E1EEE9}
[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"]{background:#fff}
[data-testid="stMarkdownContainer"] table{width:100%;border-collapse:separate;border-spacing:0;overflow:hidden;border:1px solid var(--line);border-radius:12px;background:#fff}
[data-testid="stMarkdownContainer"] th{background:#EAF2EF;color:var(--brand);font-weight:700}
[data-testid="stMarkdownContainer"] th,[data-testid="stMarkdownContainer"] td{padding:12px 14px;border-bottom:1px solid #E8EFED;text-align:left}
[data-testid="stMarkdownContainer"] tr:last-child td{border-bottom:0}
[data-testid="stMarkdownContainer"] tr:hover td{background:#F5F8F7}
.stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"]>button{min-height:48px;padding:0 1rem;border-radius:10px;border:1px solid var(--line);font-size:17px;font-weight:700;transition:background-color .18s ease,border-color .18s ease,box-shadow .18s ease,transform .18s ease}
.stButton>button:hover,.stDownloadButton>button:hover,[data-testid="stFormSubmitButton"]>button:hover{border-color:var(--sage);box-shadow:0 4px 12px rgba(22,133,138,.12);transform:translateY(-1px)}
button[data-testid="stBaseButton-primary"]{background:var(--sage)!important;border-color:var(--sage)!important;color:#fff!important}
button[data-testid="stBaseButton-primary"]:hover{background:#126D73!important;border-color:#126D73!important}
button[data-testid="stBaseButton-secondary"]{border-color:var(--line)!important;color:var(--brand)!important;background:#fff!important}
button[data-testid="stBaseButton-secondary"]:hover{background:#EAF2EF!important}
button:focus-visible,input:focus-visible,textarea:focus-visible,[role="tab"]:focus-visible{outline:3px solid #78A99A!important;outline-offset:3px}
input:not([type="checkbox"]):not([type="radio"]),textarea,[data-baseweb="select"]>div{min-height:48px;border-radius:10px!important;border-color:var(--line)!important;font-size:18px}
[data-testid="stWidgetLabel"]{font-size:18px}
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea{background:#FBFCFB}
[data-testid="stFileUploader"],[data-testid="stAudioInput"],[data-testid="stCameraInput"]{border:1px dashed #AFC7C0;border-radius:14px;background:#F7FAF8;padding:8px}
[data-testid="stRadio"] label,[data-testid="stCheckbox"] label{min-height:48px;align-items:center}
[data-testid="stTabs"] [role="tab"]{min-height:48px;padding:0 1rem;font-size:18px;font-weight:700;color:var(--muted);transition:color .18s ease,border-color .18s ease}
[data-testid="stTabs"] [aria-selected="true"]{color:var(--brand)!important;border-bottom-color:var(--sage)!important}
[data-testid="stProgressBar"]>div>div{transition:width .4s ease;background-color:var(--sage)}
[data-testid="stExpander"],[data-testid="stForm"]{border:1px solid var(--line);border-radius:14px;background:#fff}
[data-testid="stExpander"]{box-shadow:0 4px 14px rgba(30,86,92,.035);transition:border-color .18s ease,box-shadow .18s ease}
[data-testid="stExpander"]:hover{border-color:#B8CFCA;box-shadow:0 8px 18px rgba(30,86,92,.07)}
[data-testid="stExpander"] summary{min-height:52px}
[data-testid="stForm"]{padding:18px}
[data-testid="stAlert"]{border-radius:14px;border:1px solid rgba(30,86,92,.12);box-shadow:0 4px 14px rgba(30,86,92,.035)}
[data-testid="stAlert"][kind="info"]{background:#EDF5F2;color:#234B49}
[data-testid="stAlert"][kind="success"]{background:#EAF4EF;color:#28584D}
[data-testid="stAlert"][kind="warning"]{background:#FBF5E9;color:#654E20}
[data-testid="stAlert"][kind="error"]{background:#FAEEEE;color:#6A3333}
[data-testid="stPlotlyChart"]{overflow:hidden;border:1px solid var(--line);border-radius:16px;background:#fff;padding:10px;box-shadow:0 8px 20px rgba(30,86,92,.05)}
[data-testid="stJson"]{box-shadow:0 6px 18px rgba(30,86,92,.04)}
.stMarkdown hr{border-color:var(--line)}
.nn-hero{position:relative;isolation:isolate;overflow:hidden;background:linear-gradient(116deg,#142D40 0%,#173F52 52%,#17696E 100%);color:#fff;padding:30px 36px;border:1px solid rgba(255,255,255,.12);border-radius:18px;margin-bottom:22px;box-shadow:0 12px 28px rgba(23,54,74,.12)}
.nn-hero:after{content:"";position:absolute;z-index:-1;right:7%;top:50%;width:150px;height:150px;border-radius:50%;transform:translateY(-50%);background:radial-gradient(circle at 34% 30%,rgba(212,250,250,.62) 0 2%,transparent 3%),radial-gradient(circle at 67% 38%,rgba(147,224,230,.65) 0 2%,transparent 3%),radial-gradient(circle at 52% 66%,rgba(182,174,238,.5) 0 2%,transparent 3%),radial-gradient(ellipse at 38% 36%,rgba(117,217,223,.52),transparent 30%),radial-gradient(ellipse at 65% 62%,rgba(123,114,184,.43),transparent 38%),radial-gradient(circle at 48% 48%,rgba(60,159,171,.32),rgba(22,88,103,.08) 67%,transparent 70%);box-shadow:inset -12px -10px 24px rgba(7,28,44,.24),0 0 42px rgba(83,185,196,.12)}
.nn-hero h1{color:#fff;font-size:clamp(2rem,4vw,3rem);line-height:1.15}
.nn-hero-title{display:flex;align-items:center;gap:16px;margin:0 0 12px}
.nn-hero-logo{width:72px;height:72px;flex:none}
.nn-hero-title h1{margin:0}
.nn-hero p{position:relative;color:#E3EFEC;font-size:18px;max-width:720px}
.nn-muted{color:var(--muted);font-size:16px;line-height:1.55}
.nn-module-title{margin:0;color:var(--brand);font-size:1.05rem;line-height:1.35}
.nn-pill{display:inline-flex;align-items:center;min-height:32px;background:rgba(220,247,247,.12);border:1px solid rgba(220,247,247,.22);color:#D8F4F3;border-radius:99px;padding:4px 12px;font-size:14px;font-weight:700;letter-spacing:.04em}
.nn-lucide{display:inline-block;vertical-align:-6px;width:23px;height:23px;margin-right:8px;stroke:#16858A;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;fill:none}
[data-testid="stJson"]{border:1px solid var(--line);border-radius:14px;background:#fff;padding:12px}
@media(max-width:700px){.block-container{padding-top:1rem;padding-bottom:2rem}.nn-hero{padding:24px 22px}.nn-hero:after{right:-32px;top:auto;bottom:-45px;width:100px;height:100px}.stButton>button,.stDownloadButton>button{width:100%}[data-testid="stHeadingWithActionElements"]:has(> h1):not(.nn-hero [data-testid="stHeadingWithActionElements"]){padding:10px 14px;margin:10px 0 14px}[data-testid="stDataFrame"],[data-testid="stTable"]{border-radius:12px}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{scroll-behavior:auto!important;animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}}
@page{size:A4;margin:16mm}
@media print{
  html,body,.stApp{background:#fff!important;color:#111!important;font-size:12pt!important}
  [data-testid="stSidebar"],[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"],[data-testid="stTabs"] [role="tablist"],.stButton,.stDownloadButton,footer{display:none!important}
  .block-container{max-width:none!important;padding:0!important;margin:0!important}
  [data-testid="stMain"]{background:#fff!important}
  [data-testid="stJson"]{border:1px solid #cbd5d3!important;box-shadow:none!important;break-inside:avoid}
  [data-testid="stAlert"]{background:#fff!important;box-shadow:none!important}
  .nn-hero{background:#1E565C!important;print-color-adjust:exact;-webkit-print-color-adjust:exact;box-shadow:none!important}
  a{color:#111!important;text-decoration:none!important}
}
</style>''',unsafe_allow_html=True)

# Auth entry
if not st.session_state.user:
    logo_path=Path(__file__).resolve().parent/'assets'/'neuronexus-logo.svg'
    logo_svg=logo_path.read_text(encoding='utf-8').split('<text',1)[0].replace(
        'viewBox="0 0 520 120"',
        'viewBox="0 0 120 120"',
    )+'</svg>'
    logo_data=base64.b64encode(logo_svg.encode('utf-8')).decode('ascii')
    st.markdown(
        f'<div class="nn-hero"><div class="nn-hero-title">'
        f'<img class="nn-hero-logo" src="data:image/svg+xml;base64,{logo_data}" alt="">'
        '<h1>NeuroNexus</h1></div>'
        '<p>A calmer way to explore voice, cognition and movement measurements.</p>'
        '<span class="nn-pill">RESEARCH PROTOTYPE</span></div>',
        unsafe_allow_html=True,
    )
    left,right=st.columns([1.05,1])
    with left:
        st.subheader('Welcome')
        st.write('Sign in to access your personal research dashboard, assessments and reports.')
        st.info('NeuroNexus is experimental software. It cannot diagnose or rule out Alzheimer’s or Parkinson’s disease.')
        with st.expander('Privacy and data handling'):
            st.markdown(PRIVACY_NOTICE)
    with right:
        with st.container(border=True):
            st.subheader('Hackathon demo')
            st.caption(
                'Enter an isolated Demo Mode participant with synthetic history. '
                'No account is created and demo activity is not saved to the database.'
            )
            if st.button(
                'Enter Demo Mode',
                type='primary',
                key='enter_demo_mode',
                width='stretch',
            ):
                enter_demo_mode(st.session_state)
                st.rerun()
        login_tab,reg_tab=st.tabs(['Sign in','Create account'])
        with login_tab:
            with st.form('login_form'):
                email=st.text_input('Email address'); password=st.text_input('Password',type='password')
                submitted=st.form_submit_button('Sign in',type='primary',width='stretch')
            if submitted:
                init_db()
                user=authenticate(email.strip(),password)
                if user:
                    st.session_state.user=user
                    st.session_state.demo_mode=False
                    st.session_state.pop('demo_sessions',None)
                    st.rerun()
                else: st.error('Email or password is incorrect.')
        with reg_tab:
            with st.form('register_form'):
                name=st.text_input('Full name',max_chars=120); email2=st.text_input('Email address',max_chars=254); pw=st.text_input('Create password',type='password'); pw2=st.text_input('Confirm password',type='password')
                account_role=st.selectbox('Account type',['Individual','Caregiver'])
                caregiver=st.text_input('Caregiver email (optional; for individual accounts)',max_chars=254)
                ok=st.checkbox('I understand this is a research prototype, not a medical diagnostic service.')
                create=st.form_submit_button('Create account',type='primary',width='stretch')
            if create:
                valid_email=bool(re.fullmatch(r'[^@\s]+@[^@\s]+\.[^@\s]+',email2.strip()))
                valid_caregiver_email=(
                    not caregiver.strip()
                    or bool(re.fullmatch(r'[^@\s]+@[^@\s]+\.[^@\s]+',caregiver.strip()))
                )
                if (
                    not name.strip()
                    or not valid_email
                    or not valid_caregiver_email
                    or len(pw)<10
                    or pw!=pw2
                    or not ok
                ):
                    st.error('Enter a name and valid email address, use a password of at least 10 characters, and accept the notice.')
                else:
                    try:
                        init_db()
                        register(
                            name,
                            email2,
                            pw,
                            'caregiver' if account_role=='Caregiver' else 'individual',
                            caregiver,
                            consent_flags={"registration_acknowledged": True},
                        )
                        st.success('Account created. Please sign in.')
                    except Exception as exc:
                        if is_duplicate_email_error(exc):
                            st.error('That email may already be registered.')
                        else:
                            logger.error(
                                "Account registration failed (%s).",
                                type(exc).__name__,
                            )
                            st.error(
                                'The account could not be created. Check the '
                                'form and try again later.'
                            )
    st.stop()

user=st.session_state.user
with st.sidebar:
    st.image('assets/neuronexus-logo.svg',width=190,alt='NeuroNexus logo')
    st.caption('Digital Biomarker Research OS')
    st.write(f"Signed in as **{user['name']}**")
    if user['role']=='admin': workspace_pages=['Admin overview','User directory','Research assessments','Model registry','My profile','Privacy & Responsible AI','Documentation']
    elif user['role']=='caregiver': workspace_pages=['Caregiver overview','My profile','Privacy & Responsible AI','Documentation']
    else: workspace_pages=['My dashboard','Brain health timeline','Screening assessments','My reports','Wellness coach','My profile','Privacy & Responsible AI','Documentation']
    if st.session_state.get('workspace_page') not in workspace_pages:
        st.session_state['workspace_page']=workspace_pages[0]
    page=st.radio('Workspace',workspace_pages,key='workspace_page')
    st.divider(); st.caption('Experimental research software · v2.0 prototype')
    if st.button('Sign out',width='stretch'):
        reset_session(st.session_state)
        st.session_state.reaction_times=[]
        st.session_state.reaction_started=None
        st.session_state.reaction_ready=False
        st.session_state.user=None
        st.session_state.demo_mode=False
        st.session_state.pop('demo_sessions',None)
        st.rerun()

# Shared workspace header
with st.container(border=True):
    header_title,header_status=st.columns([2.5,1])
    with header_title:
        st.markdown(f'### {LUCIDE_ICONS["health"]} NeuroNexus',unsafe_allow_html=True)
        st.caption('Understand your measurements. Track your sessions. Know when to seek professional advice.')
    with header_status:
        st.markdown('**Research prototype**')
        st.caption('Voice · cognition · movement')
if is_demo_mode(user):
    st.warning(DEMO_MODE_NOTICE,title='DEMO MODE · Synthetic participant')
st.info(DIAGNOSTIC_DISCLAIMER,title='Research-use only')

if page=='Admin overview':
    data=admin_overview(); st.title('Administration overview'); st.caption('Live account and session activity from the local demonstration database.')
    people_df=pd.DataFrame(data['people'])
    recent_df=pd.DataFrame(data['recent'])
    role_counts=people_df['role'].value_counts() if not people_df.empty else pd.Series(dtype='int64')
    caregiver_count=int(role_counts.get('caregiver',0))
    admin_count=int(role_counts.get('admin',0))
    if not recent_df.empty:
        recent_df['created_at']=pd.to_datetime(recent_df['created_at'],errors='coerce')
        recent_df=recent_df.dropna(subset=['created_at'])
        today=pd.Timestamp.now().normalize()
        sessions_today=int((recent_df['created_at'].dt.normalize()==today).sum())
        activity_by_day=(recent_df.assign(day=recent_df['created_at'].dt.strftime('%b %d'))
                         .groupby('day',sort=False).size().rename('Sessions').reset_index()
                         .rename(columns={'day':'Date'}))
    else:
        sessions_today=0
        activity_by_day=pd.DataFrame(columns=['Date','Sessions'])
    st.markdown('#### At a glance')
    metric_cols=st.columns(5)
    with metric_cols[0]:
        st.metric('Total accounts',data['users'])
        st.button('View all accounts',key='admin_all_accounts',width='stretch',
                  on_click=navigate_workspace,args=('User directory','All users'))
    with metric_cols[1]:
        st.metric('Individuals',data['individuals'])
        st.button('View individuals',key='admin_individual_accounts',width='stretch',
                  on_click=navigate_workspace,args=('User directory','Individuals'))
    with metric_cols[2]:
        st.metric('Caregivers',caregiver_count)
        st.button('View caregivers',key='admin_caregiver_accounts',width='stretch',
                  on_click=navigate_workspace,args=('User directory','Caregivers'))
    metric_cols[3].metric('Saved sessions',data['sessions'])
    metric_cols[4].metric('Sessions today · latest 30',sessions_today,help='Sessions dated today among the 30 most recent saved sessions.')
    chart_col,roles_col=st.columns([1.5,1])
    with chart_col:
        with st.container(border=True):
            st.subheader('Recent session activity')
            st.caption('Daily sessions across the latest 30 recorded activities.')
            if activity_by_day.empty:
                st.info('Session activity will appear here after users save sessions.')
            else:
                st.bar_chart(activity_by_day,x='Date',y='Sessions',alt='Daily session activity across the latest 30 recorded sessions')
    with roles_col:
        with st.container(border=True):
            st.subheader('Accounts by role')
            role_chart=pd.DataFrame({
                'Role':['Individual','Caregiver','Admin'],
                'Accounts':[int(role_counts.get('individual',0)),caregiver_count,admin_count],
            })
            st.bar_chart(role_chart,x='Role',y='Accounts',horizontal=True,alt='Account counts for individuals, caregivers, and admins')
    st.subheader('Latest session activity')
    st.caption('The newest 30 saved sessions, refreshed whenever the dashboard loads.')
    if recent_df.empty:
        st.info('No saved sessions yet.')
    else:
        recent_table=recent_df.rename(columns={'id':'Session ID','created_at':'Date and time','name':'Account name','email':'Email'})
        recent_table['Date and time']=recent_table['Date and time'].dt.strftime('%b %d, %Y · %I:%M %p')
        st.dataframe(recent_table,hide_index=True,alt='Latest saved session activity')
    st.info('This is a prototype admin view. Before real deployment, add server-side authorization, audit logs, role management, secure hosting and a formal privacy/security review.')
elif page=='User directory':
    data=admin_overview(); st.title('User directory')
    people_df=pd.DataFrame(data['people'])
    role_counts=people_df['role'].value_counts() if not people_df.empty else pd.Series(dtype='int64')
    caregiver_count=int(role_counts.get('caregiver',0))
    admin_count=int(role_counts.get('admin',0))
    st.caption('Search accounts and view individuals, caregivers, and administrators separately.')
    search_col,role_col,sort_col=st.columns([1.5,1.1,1.2])
    with search_col:
        directory_search=st.text_input('Search accounts',placeholder='Name, email, or account ID',key='directory_search')
    with role_col:
        selected_role=st.segmented_control('Account type',['All users','Individuals','Caregivers','Admins'],default='All users',key='directory_role')
    with sort_col:
        sort_order=st.selectbox('Sort users by',['Newest first','Oldest first','Name A–Z','Name Z–A','ID ascending','ID descending'],key='directory_sort')
    if selected_role!='All users':
        role_value={'Individuals':'individual','Caregivers':'caregiver','Admins':'admin'}[selected_role]
        people_df=people_df[people_df['role'].eq(role_value)]
    if directory_search.strip():
        query=directory_search.strip().casefold()
        search_match=(
            people_df['name'].fillna('').astype(str).str.casefold().str.contains(query,regex=False)
            | people_df['email'].fillna('').astype(str).str.casefold().str.contains(query,regex=False)
            | people_df['id'].astype(str).str.contains(query,regex=False)
        )
        people_df=people_df[search_match]
    sort_options={
        'Newest first':('created_at',False),
        'Oldest first':('created_at',True),
        'Name A–Z':('name',True),
        'Name Z–A':('name',False),
        'ID ascending':('id',True),
        'ID descending':('id',False),
    }
    sort_column,ascending=sort_options[sort_order]
    if not people_df.empty:
        people_df=people_df.sort_values(sort_column,ascending=ascending,kind='stable')
    result_col,role_summary_col=st.columns([1,1])
    with result_col:
        st.metric('Accounts shown',len(people_df),help='Updates as the search, role, and sort controls change.')
    with role_summary_col:
        st.caption(f"Directory total: {len(data['people'])} accounts · {caregiver_count} caregivers · {int(role_counts.get('individual',0))} individuals · {admin_count} admins")
    if people_df.empty:
        st.info('No accounts match your search and account-type filters.')
    else:
        directory_table=people_df.rename(columns={'id':'Account ID','name':'Name','email':'Email','role':'Account type','created_at':'Joined'})
        st.dataframe(directory_table,hide_index=True,alt='Searchable user directory with account types and join dates')
elif page=='Model registry':
    st.title('Research model registry')
    st.caption('Model versions, dataset sources, grouped-CV metrics and validation stage per disease pathway. Research administration only — never a clinical claim.')
    reg=get_registry()
    for disease,info in reg.items():
        st.subheader('Parkinson’s pathway' if disease=='parkinsons' else 'Alzheimer’s pathway')
        if info['available']:
            m=info['subject_level']
            cols=st.columns(4)
            cols[0].metric('Subject-level balanced acc',f"{m['balanced_accuracy']:.3f}")
            cols[1].metric('Sensitivity',f"{m['sensitivity']:.3f}")
            cols[2].metric('Specificity',f"{m['specificity']:.3f}")
            cols[3].metric('Validation stage',"2 of 4 (held-out participants)")
            st.caption(f"Model: {info['model_name']} v{info.get('version')} · Selected: {info['selected_model']} · Participants: {info['participants']} · Dataset: {info['dataset_source']}")
            if info.get('models_compared'):
                with st.expander('Models compared (balanced accuracy)'):
                    st.json(info['models_compared'])
            confidence_interval=info.get('subject_level_95ci')
            if confidence_interval and len(confidence_interval)==2:
                st.caption(f"95% CI (subject-level): {confidence_interval[0]:.3f} – {confidence_interval[1]:.3f} · Trained: {info.get('trained_on')}")
            else:
                st.caption(f"Trained: {info.get('trained_on') or 'Not reported'}")
            for lim in info['limitations']: st.warning(lim)
        else:
            st.info(info['message'])
    st.divider()
    st.markdown('**Validation ladder:** 1 = model trained · 2 = evaluated on held-out participants · 3 = externally validated · 4 = clinically validated. NeuroNexus models are at most stage 2.')
elif page=='Research assessments':
    st.title('Research model laboratory')
    st.write('Develop and evaluate disease-specific research models using appropriately labelled datasets. This workspace is for research administration, not patient diagnosis.')
    st.subheader('Parkinson’s voice baseline — first research model')
    st.markdown('Upload the **UCI Oxford Parkinson’s voice CSV** containing `name`, numeric voice-feature columns, and `status` (0 = control, 1 = Parkinson’s-labelled). The training pipeline keeps recordings from the same participant together during cross-validation.')
    st.info('Important: this dataset contains precomputed biomedical voice features. It is not compatible with the app’s current MFCC summary, phone accelerometer or gait features. Training this baseline does not enable live voice or multimodal patient predictions.')
    dataset=st.file_uploader('Upload labelled Parkinson’s research CSV',type=['csv'],key='pd_research_dataset')
    if st.button('Train and evaluate research baseline',type='primary',disabled=dataset is None):
        try:
            result=train_parkinson_voice_model(dataset,output_dir='models')
            st.session_state['pd_model_metadata']=result
            st.success('Research baseline trained and evaluated. Review limitations before interpreting any metric.')
        except Exception as exc:
            logger.error(
                "Research dataset analysis failed (%s).",
                type(exc).__name__,
            )
            st.error(
                'The research dataset could not be processed. Check its format '
                'and required columns, then retry.'
            )
    result=st.session_state.get('pd_model_metadata')
    if result:
        st.subheader('Participant-grouped evaluation')
        st.caption(f"{result['dataset_rows']} recordings · {result['participant_count']} participants · {result['feature_count']} voice features")
        metrics=result['metrics']
        cols=st.columns(4)
        for col,(label,key) in zip(cols,[('Balanced accuracy','balanced_accuracy'),('Sensitivity','sensitivity'),('Specificity','specificity'),('F1 score','f1')]):
            value=metrics.get(key); col.metric(label, f"{value:.3f}" if value is not None else 'N/A')
        st.json(metrics['confusion_matrix'])
        st.warning('These are internal cross-validation results on a small, retrospective voice dataset—not clinical performance and not evidence that this model can identify disease in an individual. Independent external validation is required.')
        st.download_button('Download model evaluation metadata',json.dumps(result,indent=2),file_name='parkinson_voice_research_metadata.json',mime='application/json')
elif page=='Caregiver overview':
    st.title('Caregiver portal'); st.write('View session summaries only for participants who entered your email as their caregiver contact.')
    rows=caregiver_sessions(user['email'])
    if rows:
        participant_emails=sorted({row['participant_email'] for row in rows})
        participant_names={
            row['participant_email']:row['participant_name'] for row in rows
        }
        selected_email=st.selectbox(
            'Participant timeline',
            participant_emails,
            format_func=lambda email:f"{participant_names[email]} · {email}",
            key='caregiver_timeline_participant',
        )
        render_brain_health_timeline(
            [row for row in rows if row['participant_email']==selected_email],
            'caregiver',
            participant_name=participant_names[selected_email],
        )
    elif user['email']=='caregiver@neuronexus.local':
        st.info(
            'This demo timeline is synthetic and belongs to the demo account; it '
            'does not represent a linked participant.'
        )
        render_brain_health_timeline(
            sessions_for_user(user,st.session_state),
            'caregiver_demo',
            participant_name='Demo Participant',
        )
    if rows:
        for r in rows:
            with st.expander(f"{r['participant_name']} · {r['created_at']}"):
                st.json(json.loads(r['summary']))
    else: st.info('No linked participant sessions yet. Ask the participant to add this email in My profile. No automatic disease alerts are generated.')
elif page=='My profile':
    st.title('My profile')
    if is_demo_mode(user):
        st.info(DEMO_MODE_NOTICE,title='Demo profile · no persistent account')
        st.caption(
            'Profile editing, saved-account history, and account deletion are '
            'unavailable in Demo Mode.'
        )
        st.button(
            'Reset demo data',
            type='secondary',
            on_click=reset_demo_mode,
            args=(st.session_state,),
        )
        st.stop()
    role_options=['Individual (Participant)','Caregiver','Admin / Clinician Triage']
    actual_preview={'individual':role_options[0],'caregiver':role_options[1],'admin':role_options[2]}
    role_preview=st.radio(
        'Preview role',
        role_options,
        index=role_options.index(actual_preview.get(user['role'],role_options[0])),
        horizontal=True,
        key=f"profile_role_preview_{user['id']}",
    )
    st.caption('Preview only: this switcher does not change your signed-in role, permissions, or access to another person’s data.')
    profile_generation=int(st.session_state.get('profile_widget_generation',0))
    profile_notice=st.session_state.pop('profile_notice',None)
    if profile_notice:
        st.success(profile_notice)
    if not st.session_state.get('consent_granted'):
        has_optional_context=bool(
            st.session_state.get('baseline_context')
            or st.session_state.get('caregiver_observation_notes')
            or st.session_state.get('caregiver_notes_consent')
        )
        st.session_state.baseline_context=None
        st.session_state.caregiver_observation_notes=[]
        st.session_state.caregiver_notes_consent=False
        if has_optional_context:
            profile_generation+=1
            st.session_state.profile_widget_generation=profile_generation

    if role_preview==role_options[0]:
        st.subheader('Baseline calibration')
        baseline=st.session_state.get('baseline_context') or {}
        if not st.session_state.get('consent_granted'):
            st.info('Grant consent on Screening assessments before saving optional profile context.')
        age_options=['Not provided','18–39','40–59','60–69','70+']
        sex_options=['Not provided','Female','Male','Intersex','Prefer not to say']
        hand_options=['Not provided','Right','Left','Ambidextrous','Prefer not to say']
        device_options=['Not provided','Touchscreen / Stylus','Mouse / Trackpad']
        def baseline_index(field,options):
            return options.index(baseline[field]) if baseline.get(field) in options else 0
        with st.form(f'baseline_context_form_{profile_generation}'):
            age=st.selectbox('Age bracket',age_options,index=baseline_index('age_bracket',age_options))
            biological_sex=st.selectbox('Biological sex',sex_options,index=baseline_index('biological_sex',sex_options))
            dominant_hand=st.selectbox('Dominant hand',hand_options,index=baseline_index('dominant_hand',hand_options))
            input_device=st.selectbox('Primary input device',device_options,index=baseline_index('primary_input_device',device_options))
            save_baseline=st.form_submit_button(
                'Save baseline context',
                type='primary',
                disabled=not st.session_state.get('consent_granted'),
            )
        if save_baseline:
            st.session_state.baseline_context={
                'age_bracket':age,
                'biological_sex':biological_sex,
                'dominant_hand':dominant_hand,
                'primary_input_device':input_device,
            }
            st.success('Optional context saved in this session. It appears in reports generated from this session.')

        st.subheader('Privacy & data governance')
        if st.session_state.get('consent_granted'):
            st.success('Consent status: Granted for the current session')
        else:
            st.info('Consent status: Not granted for the current session')
        history_key=f'local_history_enabled_{profile_generation}'
        if history_key not in st.session_state:
            st.session_state[history_key]=bool(st.session_state.get('local_history_enabled',False))
        history_enabled=st.toggle('Enable saved session history',key=history_key)
        st.session_state.local_history_enabled=history_enabled
        st.caption(
            'When enabled, you may save derived reports to the configured persistent '
            'store from the Report tab. Turning this off does not delete saved reports; '
            'use Purge My Session Data to delete them.'
        )
        if st.button('Purge My Session Data',type='secondary'):
            deleted_sessions=delete_user_sessions(user['id'])
            reset_session(st.session_state)
            st.session_state.reaction_times=[]
            st.session_state.reaction_started=None
            st.session_state.reaction_ready=False
            st.session_state.profile_notice=(
                f'Purged current session data and {deleted_sessions} saved session '
                f'{"summary" if deleted_sessions == 1 else "summaries"}. Downloaded files remain.'
            )
            st.rerun()
        st.caption('Purge removes this account’s finalized reports, local draft, and current in-memory assessment/context data. Downloaded files, account details, and system logs are outside this prototype control.')
    elif role_preview==role_options[1]:
        st.subheader('Linked participant')
        with st.container(border=True):
            st.caption('Sample card · not linked to an actual participant')
            linked_col,relationship_col,age_col=st.columns(3)
            linked_col.metric('Participant ID','SAMPLE-P-1042')
            relationship_col.metric('Relationship','Adult child')
            age_col.metric('Age bracket','70+ · illustrative')
        st.caption('This preview does not reveal or authorize access to another person’s account.')
        assisted_key=f'assisted_session_mode_{profile_generation}'
        if assisted_key not in st.session_state:
            st.session_state[assisted_key]=bool(st.session_state.get('assisted_session_mode',False))
        assisted=st.toggle('Assisted Session Mode',key=assisted_key)
        st.session_state.assisted_session_mode=assisted
        st.subheader('Caregiver observation notes')
        if not st.session_state.get('consent_granted'):
            st.info('Participant session consent is required before caregiver notes can be added to a report.')
        notes_consent_key=f'caregiver_notes_consent_{profile_generation}'
        if notes_consent_key not in st.session_state:
            st.session_state[notes_consent_key]=bool(st.session_state.get('caregiver_notes_consent',False))
        notes_consent=st.checkbox(
            'Participant agrees these caregiver observations may be included in this session report',
            key=notes_consent_key,
            disabled=not st.session_state.get('consent_granted'),
        )
        prior_notes_consent=bool(st.session_state.get('caregiver_notes_consent',False))
        st.session_state.caregiver_notes_consent=bool(notes_consent and st.session_state.get('consent_granted'))
        if (
            st.session_state.caregiver_notes_consent
            and not (user.get("consent_flags") or {}).get("caregiver_notes", False)
        ):
            updated_flags=update_consent_flag(user["id"],"caregiver_notes",True)
            user={**user,"consent_flags":updated_flags}
            st.session_state.user=user
        if prior_notes_consent and not st.session_state.caregiver_notes_consent:
            st.session_state.caregiver_observation_notes=[]
            st.session_state.profile_widget_generation=profile_generation+1
            profile_generation=int(st.session_state.profile_widget_generation)
        note_options=[
            ('tremor','Observed tremor'),
            ('memory','Observed memory lapses'),
            ('gait','Observed gait changes'),
        ]
        if st.session_state.caregiver_notes_consent:
            selected_notes=[]
            for note_key,label in note_options:
                widget_key=f'caregiver_note_{note_key}_{profile_generation}'
                checked=st.checkbox(label,key=widget_key)
                if checked:
                    selected_notes.append(label)
            st.session_state.caregiver_observation_notes=selected_notes
        else:
            st.session_state.caregiver_observation_notes=[]
            st.caption('Selected notes, if any, are user-reported observations—not clinical findings—and are added to the PDF report.')
    else:
        st.subheader('De-identified triage table preview')
        st.caption('Illustrative template only; no patient records or triage decisions are shown.')
        triage_preview=pd.DataFrame([
            {'Session ID':'DEMO-S-001','Risk flag':'Not generated','Flagged biomarkers':'No validated individual flags','Referral status':'Not assigned'},
            {'Session ID':'DEMO-S-002','Risk flag':'Not generated','Flagged biomarkers':'No validated individual flags','Referral status':'Not assigned'},
            {'Session ID':'DEMO-S-003','Risk flag':'Not generated','Flagged biomarkers':'No validated individual flags','Referral status':'Not assigned'},
        ])
        st.dataframe(triage_preview,hide_index=True,alt='Synthetic triage table template with no patient risk classifications')
        st.caption('This prototype does not calculate Low, Moderate, or Elevated clinical risk levels and does not generate referrals.')

        st.subheader('Model governance & fairness')
        registry=get_registry()
        park=registry['parkinsons']
        def format_metric(metrics,key):
            value=metrics.get(key)
            return f'{value:.3f}' if isinstance(value,(int,float)) else 'Not reported'
        if park.get('available'):
            subject=park.get('subject_level') or {}
            recording=park.get('recording_level') or {}
            park_validation=(
                f"Grouped CV; subject-level balanced accuracy {format_metric(subject,'balanced_accuracy')}, "
                f"sensitivity {format_metric(subject,'sensitivity')}, specificity {format_metric(subject,'specificity')}; "
                f"recording-level balanced accuracy {format_metric(recording,'balanced_accuracy')}. Research evaluation only."
            )
            park_source=f"{park.get('dataset_source','Unknown')} · {park.get('participants','Unknown')} participants; upstream license/citation not verified."
            park_version=park.get('version') or 'Not recorded'
            park_name=park.get('model_name') or 'Parkinson voice research model'
        else:
            park_validation=park.get('message','No validation metrics available.')
            park_source='multiple_sound attribution not verified'
            park_version='Unavailable'
            park_name='Parkinson voice research model'
        alzheimer=registry['alzheimers']
        if alzheimer.get('available'):
            alzheimer_meta=alzheimer.get('metadata') or {}
            alzheimer_dataset=alzheimer_meta.get('dataset') or {}
            alzheimer_metrics=alzheimer.get('subject_level') or {}
            alzheimer_validation=(
                f"Internal grouped participant CV; accuracy {format_metric(alzheimer_metrics,'accuracy')}, "
                f"balanced accuracy {format_metric(alzheimer_metrics,'balanced_accuracy')}, "
                f"ROC-AUC {format_metric(alzheimer_metrics,'roc_auc')}, "
                f"sensitivity {format_metric(alzheimer_metrics,'sensitivity')}, "
                f"specificity {format_metric(alzheimer_metrics,'specificity')}. Not clinical validation."
            )
            alzheimer_version=alzheimer.get('version') or 'Not recorded'
            alzheimer_source=(
                f"UCI DARWIN · {alzheimer_dataset.get('license','license not recorded')} · "
                f"DOI {alzheimer_dataset.get('doi','not recorded')}"
            )
        else:
            alzheimer_validation=alzheimer.get('message','No research model is installed.')
            alzheimer_version='Unavailable'
            alzheimer_source='UCI DARWIN; model not trained'
        governance=pd.DataFrame([
            {'Pathway':park_name,'Version':park_version,'Dataset attribution':park_source,'Validation':park_validation},
            {'Pathway':"Alzheimer's kinematics (DARWIN)",'Version':alzheimer_version,'Dataset attribution':alzheimer_source,'Validation':alzheimer_validation},
            {'Pathway':"Alzheimer's speech",'Version':'No trained model','Dataset attribution':'Not applicable','Validation':'Descriptive features only; no individual inference'},
        ])
        st.dataframe(governance,hide_index=True,alt='Current model versions, dataset attributions, and research validation status')
        st.subheader('Age-group fairness audit preview · not computed')
        fairness_preview=pd.DataFrame([
            {'Age group':'18–39','Sensitivity':'Not evaluated','Specificity':'Not evaluated','Audit status':'No subgroup validation evidence'},
            {'Age group':'40–59','Sensitivity':'Not evaluated','Specificity':'Not evaluated','Audit status':'No subgroup validation evidence'},
            {'Age group':'60+','Sensitivity':'Not evaluated','Specificity':'Not evaluated','Audit status':'No subgroup validation evidence'},
        ])
        st.dataframe(fairness_preview,hide_index=True,alt='Age-group fairness audit template; subgroup metrics have not been evaluated')
        st.caption('The age-group rows are an audit template, not simulated performance results. Demographic subgroup data and fairness metrics are not available.')

    st.divider()
    profile_col,account_col=st.columns([1.25,1])
    with profile_col:
        st.subheader('Personal details')
        with st.form('profile'):
            nm=st.text_input('Full name',value=user['name'],max_chars=120)
            cg=st.text_input('Caregiver email',value=user.get('caregiver_email') or '',max_chars=254)
            save=st.form_submit_button('Save profile',type='primary')
    with account_col:
        st.subheader('Account overview')
        role_label={'individual':'Individual','caregiver':'Caregiver','admin':'Administrator'}.get(user['role'],user['role'].title())
        joined=(user.get('created_at') or 'Unknown')[:10]
        st.metric('Account type',role_label)
        st.text_input('Email address',value=user['email'],disabled=True,key=f"profile_email_{user['id']}")
        st.caption(f'Member since {joined}')
    if save:
        valid_caregiver_email=(
            not cg.strip()
            or bool(re.fullmatch(r'[^@\s]+@[^@\s]+\.[^@\s]+',cg.strip()))
        )
        if not nm.strip():
            st.error('Enter a name before saving your profile.')
        elif not valid_caregiver_email:
            st.error('Enter a valid caregiver email address or leave the field blank.')
        else:
            update_profile(user['id'],nm.strip(),cg.strip())
            st.session_state.user={
                **user,
                'name':nm.strip(),
                'caregiver_email':cg.strip(),
            }
            st.success('Profile updated.')
elif page=='Privacy & Responsible AI':
    st.title('Privacy & Responsible AI')
    st.caption(
        'An accessible summary of what NeuroNexus collects, how it is processed, '
        'where it may be stored, and what the results can and cannot tell you.'
    )
    if is_demo_mode(user):
        st.warning(
            DEMO_MODE_NOTICE,
            title='Demo mode · synthetic data',
        )
    for section_title, section_text in CONSENT_INFORMATION:
        with st.container(border=True):
            st.subheader(section_title)
            st.write(section_text)
    st.info(
        RESPONSIBLE_AI_NOTICE,
        title='Responsible AI and clinical review',
    )
    with st.expander('Technical privacy and security details'):
        st.markdown(PRIVACY_NOTICE)
        st.info(STORAGE_ARCHITECTURE_NOTE)
        st.caption(
            'PostgreSQL transport is encrypted when configured. App HTTPS depends '
            'on the hosting platform or reverse proxy; this prototype does not '
            'provide a production security guarantee.'
        )
elif page=='Documentation':
    st.title('Technical & clinical reference')
    st.caption('Implementation facts, research provenance, safety boundaries, and a proposed path to evaluation. This prototype is not approved for clinical use.')
    st.info(STORAGE_ARCHITECTURE_NOTE,title='Storage configuration and limits')
    st.caption(f'Persistent account/report backend: {storage_backend()}. Local assessment drafts remain on this app instance.')
    with st.expander('1. System architecture & multimodal pipeline',expanded=True):
        st.markdown("""
**Design principle — “To Sense is to Predict” is an aspiration, not a current clinical claim.** NeuroNexus brings several optional signal tasks into one consent-gated Streamlit session and report. It does not combine them in a validated multimodal disease-prediction model.

1. **Voice capture and standardization.** WAV, MP3, M4A, OGG, and FLAC inputs are validated (25 MB, 1–120 seconds), decoded, and converted to mono 16 kHz WAV. The general voice path displays a bounded waveform and magnitude spectrogram and reports MFCC summaries, RMS, a near-silence frame ratio, zero-crossing rate, spectral centroid, median pitch, and technical quality. The near-silence ratio is not a pause count. This path does not calculate speech rate, vocal jitter, shimmer, HNR, or an individual model output.
2. **Separate sustained-vowel research path.** A sustained vowel can be processed with Praat-compatible settings for jitter, shimmer, HNR, pitch, and voice-break features. Those 26 features are passed to the existing Parkinson’s research artifact for an experimental pattern comparison. The displayed model score is not calibrated confidence or an individual disease probability. This result is task-specific, internally evaluated, and not a diagnosis.
3. **Clock drawing.** The current mouse/touch canvas records path geometry such as stroke count, approximate path length, direction changes, and bounding-box area. It has no calibrated pressure or reliable pointer timestamps, so DARWIN-style pen velocity, air/pause time, and pressure cannot be calculated. A DARWIN kinematic model is trained for research, but this canvas schema is incompatible and is not scored by that model.
4. **Other behavioral/movement tasks.** Resting motion derives an acceleration-magnitude spectrum from a supplied CSV; gait reports candidate peaks and estimated cadence using a preliminary heuristic; reaction time is a browser/device response-time demonstration. None is a validated disease classifier.
5. **Session and report.** Derived measurements are assembled into a JSON/PDF report. Finalized reports are saved only when a user explicitly saves them. Incomplete module results stay in a local SQLite draft and are removed on reset, sign-out, or successful report save; raw audio and sensor uploads are not copied into that draft.

The speech-feature tab also computes descriptive transcript features (for example word count and type-token ratio). No trained Alzheimer’s speech classifier is installed.
""")
    with st.expander('2. Training datasets & legal governance'):
        st.markdown("""
### Parkinson’s voice artifact currently in `models/`

The checked-in metadata identifies the training source as **`multiple_sound`**: 1,040 recordings, 40 participants, 26 acoustic features, with 20 participants in each label group. It records a soft-voting **Logistic Regression + Gradient Boosting** model. The candidate pipelines use median imputation and `SelectKBest(f_classif, k=8)` feature selection; Logistic Regression and SVM pipelines standardize selected features. Evaluation uses 10-fold stratified grouped cross-validation, keeping participant recordings together.

- Participant-level grouped-CV mean: balanced accuracy **0.617**, sensitivity **0.600**, specificity **0.633**; reported 95% CI for balanced accuracy **0.443–0.790**.
- Recording-level out-of-fold metrics: ROC-AUC **0.573**, balanced accuracy **0.552**, sensitivity **0.546**, specificity **0.558**.
- These are internal retrospective research metrics, not external or clinical validation. The artifact has no recorded training date. The result must not be presented as a personal risk estimate.

The training utility also has a loader for the **UCI Oxford Parkinson’s voice CSV**, but that is not the dataset recorded in the checked-in artifact metadata. Source: [UCI Oxford Parkinson’s Disease Detection Dataset](https://archive.ics.uci.edu/dataset/174/parkinsons), DOI [10.24432/C59C74](https://doi.org/10.24432/C59C74).

### DARWIN handwriting dataset

The supplied UCI **DARWIN** table has 174 participants, an `ID`, 450 task features, and a `class` label. It is licensed **CC BY 4.0**. Attribution: Fontanella, F. (2022), *DARWIN* [Dataset], UCI Machine Learning Repository, DOI [10.24432/C55D0K](https://doi.org/10.24432/C55D0K); see the [UCI dataset record](https://archive.ics.uci.edu/dataset/732/darwin). NeuroNexus trains a calibrated logistic-regression research artifact with `SelectKBest(f_classif, k=20)` and participant-grouped cross-validation. Its metrics are internal research results, not external or clinical validation. The app does not apply the artifact to live web-canvas data.

### Attribution status and compliance

The checked-in Parkinson model metadata and local integration notes do not preserve an upstream URL, DOI, or license for the `multiple_sound` training files. The code describes them as Multiple Sound Recording data, but open-access status and redistribution rights have **not** been verified here. Do not claim complete licensing compliance or redistribute those data until the exact source and terms are recorded. Keep a dataset manifest with source, version, license, citation, access date, permitted uses, and transformations for every future training run.
""")
    with st.expander('3. Clinical scope, assumptions & limitations'):
        st.markdown("""
**Current scope:** experimental signal capture and descriptive research measurements only. NeuroNexus is **not currently a validated screening, triage, or diagnostic aid** and must not determine referral priority, disease status, treatment, or reassurance that disease is absent.

- The Parkinson’s artifact is trained on a small retrospective dataset and evaluated only with internal grouped cross-validation. It is not externally validated, calibrated for clinical risk, or validated for this app’s ordinary voice/MFCC input.
- A DARWIN kinematic research artifact is trained and registered, but it is not deployed for live canvas scoring or individual prediction. DARWIN’s tablet-derived 25-task features are not reconstructable from this mouse/touch canvas.
- Voice measures vary with language, accent, age, sex, task, microphone, room acoustics, and unrelated health conditions. Current representation and subgroup performance have not been comprehensively assessed.
- Phone motion depends on sensor calibration, placement, orientation, sampling rate, and timestamp quality; absent timestamps are assumed to be 50 Hz. The gait peak method may miss or double-count steps.
- Canvas geometry varies with mouse, touchscreen, stylus, browser, viewport, and drawing surface. This canvas has no calibrated pressure or capture-time data.
- Browser/device scheduling affects reaction-time estimates. Technical signal-quality scores measure usability of a recording, not health.

Hospital use would require a defined intended use, prospective ethics-approved studies, representative multi-site/device/language cohorts, locked analysis protocols, external validation, calibration, clinical oversight, regulatory assessment, and monitored post-deployment performance.
""")
    with st.expander('4. Privacy, security & responsible AI'):
        st.markdown("""
### Implemented safeguards and current behavior

- A consent checkbox and separate Continue action gate assessment controls. This is not a validated e-consent workflow; only a consent flag, not a signed receipt or timestamp, is stored.
- The UI tells participants not to say names or addresses. There is **no automatic PII detection or redaction**; de-identify transcripts and avoid identifiers in audio and uploads.
- Audio and sensor data are processed in the active session. Use **Clear uploaded inputs and retained audio** to clear file inputs, transcript input, and the converted audio copy while keeping completed measurements. Optional speech-audio extraction writes a temporary WAV that is removed when processing exits.
- Signed-in unfinished derived results are kept in a local SQLite draft after consent; raw uploads are not saved there. Clear-session, sign-out, and successful report-save remove the draft.
- Account names, emails, caregiver links, roles, consent flags, password salts/hashes, and explicitly saved derived reports use PostgreSQL when `DATABASE_URL` is configured, otherwise local SQLite. PostgreSQL transport requires SSL; `require` encrypts transport but does not validate the server certificate. Local SQLite is not encrypted at rest.
- The app does not load an analytics tag or remote fonts, and Streamlit usage statistics are disabled. Hosting HTTPS, access logs, backups, and platform telemetry remain outside these app controls.
- Passwords are PBKDF2-hashed, and database queries are parameterized. Caregiver report access matches an email entered in a participant profile but does not verify address ownership. Authentication and role checks are demonstration-only, not production security; review server-side access policies, retention, and audit controls before real deployment.

### Responsible AI status

Displayed feature values and technical-quality checks are **not** model explanations. SHAP/LIME, feature-attribution reporting, validated reference ranges, and subgroup fairness analysis are not implemented. No validated healthy baseline or automated “elevated” flag is available. Model metrics shown here are study-level validation summaries, not individual predictions.

Before any real participant data: complete ethics/privacy review, lawful basis and consent design, data minimization, access controls, encryption/key management, retention/deletion policy, audit logging, incident response, vendor/hosting review, and jurisdiction-specific legal review.
""")
    with st.expander('5. Real-world clinical deployment pathway'):
        st.markdown("""
**A three-minute pre-visit workflow is a hypothesis to test, not a measured capability or deployment claim.** Abu Dhabi primary-care clinics and community screening networks are possible settings for co-design; no clinical partner, approval, or deployment is currently confirmed.

1. **Prototype governance:** confirm dataset licenses and citations, intended use, ethics pathway, privacy impact, security architecture, languages, accessibility, and stakeholder requirements.
2. **Prospective silent-mode study:** recruit representative participants across sites and devices; collect reference assessments under an approved protocol; keep outputs hidden from care decisions while evaluating completion time, data quality, subgroup performance, calibration, and false-positive/false-negative impact.
3. **Clinician-supervised feasibility pilot:** only after independent evidence and required approvals, test workflow usability with trained staff. A qualified clinician remains responsible; no automated referral denial, diagnosis, or treatment decision.
4. **Controlled integration:** if evidence and regulation support it, add secure identity/access management, encrypted transport/storage, auditable consent, minimum-necessary EHR integration, retention controls, training, incident handling, drift monitoring, and a rollback plan.
5. **Ongoing oversight:** review performance across demographic/language/device groups, collect clinician and participant feedback, monitor harms and operational burden, and revalidate material model or workflow changes.

The prototype does not connect to an EHR, generate clinic referrals, or establish that a three-minute screen improves outcomes. Those claims require measured prospective evidence.
""")
elif page=='My dashboard':
    st.title(f"Welcome back, {user['name'].split()[0]}")
    st.caption('A clear view of your saved activity, available assessments, and reports.')
    sessions=sessions_for_user(user,st.session_state)
    if is_demo_mode(user):
        if st.session_state.pop('demo_reset_notice',False):
            st.success('Demo data has been reset to the initial synthetic history.')
        st.button(
            'Reset demo data',
            type='secondary',
            on_click=reset_demo_mode,
            args=(st.session_state,),
        )
    latest_report=json.loads(sessions[0]['summary']) if sessions else {}
    latest_modules=completed_module_explanations(latest_report)
    a,b,c,d=st.columns(4)
    a.metric('Saved sessions',len(sessions))
    b.metric('Assessments available',7)
    c.metric('Activities in latest session',len(latest_modules))
    d.metric('Latest session',sessions[0]['created_at'][:10] if sessions else 'Not yet')
    with st.container(border=True):
        st.subheader('Start a screening session')
        st.caption('Choose only the activities that feel comfortable. You can stop at any time.')
        st.button(
            'Start screening',
            type='primary',
            on_click=navigate_workspace,
            args=('Screening assessments',),
        )
    st.subheader('Your assessment journey')
    st.button(
        'Open brain health timeline',
        on_click=navigate_workspace,
        args=('Brain health timeline',),
        key='dashboard_open_timeline',
    )
    cols=st.columns(3)
    for col,(title,desc,icon) in zip(cols,[
        ('Voice','Speech and acoustic measurements','voice'),
        ('Cognition','Clock drawing, reaction time, and memory assessment','cognition'),
        ('Movement','Resting motion and gait measurements','movement'),
    ]):
        with col.container(border=True):
            st.markdown(
                f'<h4 class="nn-module-title">{LUCIDE_ICONS[icon]} {title}</h4>',
                unsafe_allow_html=True,
            )
            st.caption(desc)
    render_brain_health_timeline(
        sessions,
        'dashboard',
        participant_name=user['name'],
        show_start_button=True,
        wellness_plan=saved_wellness_report_context(user),
    )
    with st.container(border=True):
        st.subheader('Saved assessment activity')
        st.caption('Counts show which activities appear in your saved summaries; they are not measures of health or change.')
        assessment_counts={
            'Voice':0,
            'Resting motion':0,
            'Gait':0,
            'Clock drawing':0,
            'Reaction time':0,
            'Memory check':0,
            'Speech features':0,
        }
        for session in sessions:
            summary=json.loads(session['summary'])
            for key,label in [('resting_motion','Resting motion'),('gait','Gait')]:
                if summary.get(key):
                    assessment_counts[label]+=1
            if summary.get('voice') or summary.get('voice_pattern_features'):
                assessment_counts['Voice']+=1
            cognitive=summary.get('cognitive_tasks') or {}
            if (cognitive.get('clock_drawing') or {}).get('completed'):
                assessment_counts['Clock drawing']+=1
            if (cognitive.get('reaction_time') or {}).get('trials',0):
                assessment_counts['Reaction time']+=1
            if (cognitive.get('memory_check') or {}).get('completed'):
                assessment_counts['Memory check']+=1
            if summary.get('alzheimers_speech_features'):
                assessment_counts['Speech features']+=1
        activity_df=pd.DataFrame({
            'Assessment':list(assessment_counts),
            'Saved sessions':list(assessment_counts.values()),
        })
        if any(assessment_counts.values()):
            st.bar_chart(activity_df,x='Assessment',y='Saved sessions',horizontal=True,
                         alt='Number of saved sessions containing each assessment activity')
        else:
            st.info('Assessment activity will appear here after you save a session with completed activities.')
    latest_col,privacy_col,wellness_col=st.columns(3)
    with latest_col.container(border=True):
        st.subheader('Latest saved report')
        if sessions:
            st.caption(sessions[0]['created_at'])
            st.write(f'{len(latest_modules)} activity domain(s) recorded.')
            st.button(
                'Open my reports',
                on_click=navigate_workspace,
                args=('My reports',),
                key='dashboard_open_reports',
            )
        else:
            st.info('Your latest saved report will appear here after you save a session.')
    with privacy_col.container(border=True):
        st.subheader('Privacy controls')
        st.markdown(':green[Session history is opt-in]')
        st.caption('Save derived summaries only when history is enabled. Raw audio and sensor uploads are not included in saved session reports.')
        st.button(
            'Review privacy settings',
            on_click=navigate_workspace,
            args=('My profile',),
            key='dashboard_privacy_settings',
        )
    with wellness_col.container(border=True):
        st.subheader('Wellness companion')
        st.caption('Optional, non-prescriptive habits and a checklist for a clinician visit.')
        st.button(
            'Open wellness coach',
            on_click=navigate_workspace,
            args=('Wellness coach',),
            key='dashboard_wellness_coach',
        )
    st.subheader('Recent sessions')
    if sessions:
        for s in sessions[:5]:
            with st.expander(s['created_at']):
                st.caption('Descriptive session summary — not a diagnosis or disease-risk estimate.')
                st.dataframe(
                    pd.DataFrame(key_measurement_rows(json.loads(s['summary']))),
                    hide_index=True,
                    alt='Recorded session measurements without clinical interpretation',
                )
                with st.expander('View full structured summary'):
                    st.json(json.loads(s['summary']))
    else:
        st.info('Your saved session summaries will appear here after you complete assessments and choose Save session.')
elif page=='Brain health timeline':
    st.title('Brain health timeline')
    st.caption(
        'Review saved assessment sessions, the measurements they contain, and '
        'descriptive differences between available sessions.'
    )
    try:
        with st.spinner('Loading saved assessment history...'):
            timeline_sessions = sessions_for_user(user,st.session_state)
    except Exception as exc:
        logger.error(
            "Assessment history could not be loaded (%s).",
            type(exc).__name__,
        )
        st.error('Assessment history could not be loaded. Please try again later.')
    else:
        render_brain_health_timeline(
            timeline_sessions,
            'timeline_page',
            participant_name=user['name'],
            show_start_button=True,
            wellness_plan=saved_wellness_report_context(user),
        )
elif page=='My reports':
    st.title('My reports')
    st.caption('Review saved session summaries and download a report for your records or clinician discussion.')
    sessions=sessions_for_user(user,st.session_state)
    if is_demo_mode(user):
        st.caption('Demo reports are isolated to this session and are not saved to the account database.')
    session_reports=[(session,json.loads(session['summary'])) for session in sessions]
    saved_wellness=saved_wellness_report_context(user)
    saved_modules={}
    for _,saved_report in session_reports:
        saved_modules.update(completed_module_explanations(saved_report))
    with st.container(border=True):
        st.subheader('How to read your results')
        if saved_modules:
            for label,description in saved_modules.items():
                st.markdown(f'- **{label}:** {description}')
        else:
            st.write('No completed modules are available in your saved reports yet.')
        st.caption('These are descriptive measurements, not diagnoses or disease-risk estimates.')
    render_brain_health_timeline(
        sessions,
        'reports',
        participant_name=user['name'],
        show_start_button=True,
        wellness_plan=saved_wellness,
    )
    registry=get_registry()
    st.subheader('Research model status')
    park=registry['parkinsons']
    if park['available']:
        metrics=park.get('subject_level') or {}
        metric_text=' · '.join(
            f"{label} {metrics[key]:.3f}"
            for label,key in [
                ('Balanced accuracy','balanced_accuracy'),
                ('Sensitivity','sensitivity'),
                ('Specificity','specificity'),
            ]
            if metrics.get(key) is not None
        )
        park_status=f"Research artifact installed · internal grouped CV: {metric_text or 'metrics unavailable'} · not clinically validated."
    else:
        park_status=park.get('message','No research model is installed.')
    alzheimer_status=registry['alzheimers']
    if alzheimer_status.get('available'):
        metrics=alzheimer_status.get('subject_level') or {}
        metric_text=' · '.join(
            f"{label} {metrics[key]:.3f}"
            for label,key in [
                ('Accuracy','accuracy'),
                ('Balanced accuracy','balanced_accuracy'),
                ('Sensitivity','sensitivity'),
                ('Specificity','specificity'),
                ('ROC-AUC','roc_auc'),
            ]
            if metrics.get(key) is not None
        )
        alzheimer_model_status=(
            f"Research artifact installed · internal participant-grouped CV: "
            f"{metric_text or 'metrics unavailable'} · not clinically validated; "
            "not used for live canvas scoring."
        )
    else:
        alzheimer_model_status=alzheimer_status.get('message','No DARWIN research model is installed.')
    model_rows=[
        {'Pathway':'Parkinson’s voice','Status':park_status},
        {'Pathway':'Alzheimer’s kinematics (DARWIN)','Status':alzheimer_model_status},
        {'Pathway':'Alzheimer’s speech features','Status':'Descriptive transcript features only; no speech model is installed.'},
    ]
    st.dataframe(pd.DataFrame(model_rows),hide_index=True,alt='Research model availability and validation status; not a personal risk result')
    st.caption('Model metrics describe research validation only. They are not patient-specific predictions. The current web canvas does not supply the DARWIN 25-task tablet feature schema.')
    if not sessions: st.info('No saved reports yet. Complete assessments and save a session first.')
    for s,report in session_reports:
        with st.expander(f"Session · {s['created_at']}"):
            st.caption(f"{len(completed_module_explanations(report))} activity domain(s) recorded. Measurements are descriptive, not diagnostic.")
            st.dataframe(
                pd.DataFrame(key_measurement_rows(report)),
                hide_index=True,
                alt='Recorded measurements with no clinical reference ranges or risk interpretation',
            )
            with st.expander('View full structured report'):
                st.json(report)
            st.download_button('Download JSON',s['summary'],file_name=f"neuronexus_session_{s['id']}.json",mime='application/json',key=f"json{s['id']}")
            st.download_button(
                'Download PDF',
                make_pdf(
                    user['name'],
                    report,
                    registry,
                    session_id=s['id'],
                    session_created_at=s['created_at'],
                    history=sessions,
                    wellness_plan=saved_wellness,
                ),
                file_name=f"neuronexus_session_{s['id']}.pdf",
                mime='application/pdf',
                key=f"pdf{s['id']}",
            )
elif page=='Wellness coach':
    render_wellness_coach(
        sessions_for_user(user,st.session_state),
        'wellness',
        user,
    )
    st.button(
        'Return to dashboard',
        on_click=navigate_workspace,
        args=('My dashboard',),
    )
elif page=='Screening assessments':
    st.title('Your screening assessments'); st.write('Complete only the activities you feel comfortable doing. You may stop at any time.')
    if not st.session_state.consent_granted:
        with st.container(border=True):
            st.subheader('Before you begin: consent and safety')
            st.caption(
                'Read the information below before choosing whether to continue. '
                'Each assessment activity is optional.'
            )
            if is_demo_mode(user):
                st.warning(DEMO_MODE_NOTICE,title='Demo mode · synthetic data')
            for section_title, section_text in CONSENT_INFORMATION:
                st.markdown(f'**{section_title}**')
                st.write(section_text)
            st.info(
                RESPONSIBLE_AI_NOTICE,
                title='Research prototype · not a diagnosis',
            )
            st.checkbox(
                CONSENT_TEXT,
                key='assessment_consent_checked',
            )
            consent_col, exit_col = st.columns(2)
            continue_assessment = consent_col.button(
                'Continue to assessment',
                type='primary',
                disabled=not st.session_state.get(
                    'assessment_consent_checked', False
                ),
                width='stretch',
            )
            exit_col.button(
                'Not now — return to dashboard',
                on_click=decline_assessment_consent,
                width='stretch',
            )
        if continue_assessment:
            st.session_state.consent_granted = True
        else:
            st.stop()
    if st.session_state.get('screening_mode') not in SUPPORTED_MODES:
        st.session_state.screening_mode = SUPPORTED_MODES[0]
    st.selectbox(
        'Screening type',
        SUPPORTED_MODES,
        key='screening_mode',
        help='This identifies the workflow context in the report; it does not change the analysis.',
    )
    st.caption('The selected screening type labels the report only; it does not change measurements or model behavior.')
    if not (user.get("consent_flags") or {}).get("assessment", False):
        if is_demo_mode(user):
            updated_flags={**(user.get("consent_flags") or {}),"assessment":True}
        else:
            updated_flags=update_consent_flag(user["id"],"assessment",True)
        user={**user,"consent_flags":updated_flags}
        st.session_state.user=user
    if (
        not st.session_state.get("draft_resumed", False)
        and not has_assessment_progress(st.session_state)
    ):
        saved_draft=None if is_demo_mode(user) else load_draft(user["id"])
        if saved_draft:
            completed_modules=sum(assessment_completion(saved_draft).values())
            with st.container(border=True):
                st.subheader('Unfinished assessment draft')
                st.info(
                    f'A local draft is available with {completed_modules} completed '
                    'module(s). Raw audio and sensor uploads are not stored in the draft.'
                )
                resume_col,begin_col=st.columns(2)
                if resume_col.button(
                    'Resume Where You Left Off',
                    type='primary',
                    width='stretch',
                ):
                    restore_assessment_draft(st.session_state,saved_draft)
                    st.session_state.draft_resumed=True
                    st.session_state.draft_finalized=False
                    st.rerun()
                if begin_col.button(
                    'Start From Beginning',
                    type='secondary',
                    width='stretch',
                ):
                    reset_session(st.session_state, reset_consent=False)
                    st.session_state.reaction_times=[]
                    st.session_state.reaction_started=None
                    st.session_state.reaction_ready=False
                    st.rerun()
    completion=assessment_completion(st.session_state)
    completed_count=sum(completion.values())
    st.progress(completed_count/len(completion))
    st.caption(f'Session progress: {completed_count}/{len(completion)} modules completed')
    labels={
        'voice':'Voice',
        'resting_motion':'Resting motion',
        'gait':'Gait',
        'clock_drawing':'Clock drawing',
        'reaction_time':'Reaction time',
        'memory_check':'Memory assessment',
        'speech_features':'Speech features',
    }
    tabs=st.tabs([
        f"{'✓ ' if completion[key] else ''}{labels[key]}"
        for key in ('voice','resting_motion','gait','clock_drawing','reaction_time','memory_check','speech_features')
    ]+['Session report'])
    voice_tab,motion_tab,gait_tab,clock_tab,reaction_tab,memory_tab,alz_tab,report_tab=tabs

    with voice_tab:
        st.subheader('Voice assessment'); st.write('Read a short neutral passage or sustain a vowel for several seconds. Use the same task when repeating sessions.')
        chosen=None
        if is_demo_mode(user):
            st.caption(
                'Demo Mode uses generated sample audio only. Recording and arbitrary '
                'uploads are disabled so participant data cannot enter this demo.'
            )
            if st.button(
                'Load synthetic demo voice sample',
                key='load_demo_voice_sample',
            ):
                st.session_state.demo_voice_sample=generate_sample_vowel_wav()
                st.rerun()
            voice_bytes=st.session_state.get('demo_voice_sample')
            if voice_bytes:
                st.audio(
                    voice_bytes,
                    format='audio/wav',
                    alt='Generated synthetic voice sample for Demo Mode',
                )
                st.caption(
                    'Synthetic audio sample · the normal descriptive audio pipeline '
                    'will process it. No model score or diagnosis is generated here.'
                )
        else:
            st.caption('Avoid saying names, addresses or other identifying information.')
            st.caption(
                'WAV, MP3, M4A, OGG or FLAC · up to '
                f'{MAX_AUDIO_UPLOAD_BYTES // (1024 * 1024)} MB and 120 seconds · '
                'converted to 16 kHz mono WAV when analyzed.'
            )
            rec=st.audio_input('Record a voice sample',key=f"rec{st.session_state.upload_generation}")
            up=st.file_uploader(
                'Or upload an audio file',
                type=['wav','mp3','m4a','ogg','flac'],
                key=f"wav{st.session_state.upload_generation}",
                help=(
                    f"Maximum {MAX_AUDIO_UPLOAD_BYTES // (1024 * 1024)} MB "
                    "and 120 seconds. Input is validated before analysis."
                ),
            )
            chosen=rec or up
            voice_bytes=chosen.getvalue() if chosen is not None else None
        st.button(
            'Clear uploaded inputs and retained audio',
            on_click=clear_assessment_inputs,
            args=(st.session_state,),
            help=(
                'Clears audio, sensor-file, and matching-audio uploads, temporary '
                'transcript input, and the converted audio playback copy. Completed '
                'derived results remain; use Clear current screening session to '
                'remove those too.'
            ),
        )
        analyze_label='Analyze synthetic demo sample' if is_demo_mode(user) else 'Analyze voice'
        analysis_ran=False
        if st.button(analyze_label,disabled=voice_bytes is None):
            analysis_ran=True
            st.session_state.audio_pipeline_events=[]
            st.session_state.voice_visualization=None
            st.session_state.audio_result=None
            status=st.status('Audio received',expanded=True)
            pipeline_events=[{
                'stage':'Audio received',
                'state':'complete',
                'message':'Audio is being processed in memory for this assessment.',
            }]
            status.write('Audio received · processing is in memory.')
            stage_titles={
                'validation':'Audio validation',
                'conversion':'Audio conversion',
                'preprocessing':'Signal preprocessing',
                'feature_extraction':'Feature extraction',
                'model_inference':'Model inference',
                'results_preparation':'Results preparation',
            }

            def update_audio_progress(stage,state,message):
                title=stage_titles.get(stage,stage.replace('_',' ').capitalize())
                pipeline_events.append({
                    'stage':title,
                    'state':state,
                    'message':message,
                })
                st.session_state.audio_pipeline_events=list(pipeline_events)
                if state=='running':
                    status.update(label=f'{title} in progress',state='running',expanded=True)
                elif state=='complete':
                    status.write(f'✓ {title} · {message}')
                    status.update(label=f'{title} complete',state='running',expanded=True)
                elif state=='error':
                    status.write(f'{title} failed · {message}')
                    status.update(label=f'{title} failed',state='error',expanded=True)
                else:
                    status.write(f'{title} · {message}')

            try:
                from modules.audio_analysis import to_standard_wav

                wav_bytes,conversion=to_standard_wav(
                    voice_bytes,
                    progress_callback=update_audio_progress,
                )
                update_audio_progress(
                    'preprocessing',
                    'running',
                    'Checking the converted waveform and existing technical quality gates.',
                )
                audio_result=analyze_audio_bytes(
                    wav_bytes,
                    progress_callback=update_audio_progress,
                    source_sample_rate=conversion['source_sample_rate'],
                )
                audio_result.update({
                    'source_sample_rate':conversion['source_sample_rate'],
                    'source_channels':conversion['source_channels'],
                    'source_duration_seconds':conversion['source_seconds'],
                    'conversion_method':conversion['decoded_with'],
                })
                update_audio_progress(
                    'model_inference',
                    'skipped',
                    (
                        'No model is part of this general descriptive voice path. '
                        'The separate sustained-vowel pathway runs the existing '
                        'experimental research classifier.'
                    ),
                )
                update_audio_progress(
                    'results_preparation',
                    'running',
                    'Preparing descriptive measurements and bounded visualizations.',
                )
                visualization=None
                try:
                    visualization=audio_visualization_data(wav_bytes)
                except Exception as exc:
                    logger.error(
                        "Audio visualization generation failed (%s).",
                        type(exc).__name__,
                    )
                    st.warning(
                        'Audio measurements are available, but the waveform or '
                        'spectrogram could not be displayed.'
                    )
                st.session_state.voice_wav_bytes=wav_bytes
                st.session_state.voice_visualization=visualization
                st.session_state.audio_result=audio_result
                st.session_state.pattern_result=None
                update_audio_progress(
                    'results_preparation',
                    'complete',
                    'Audio measurements and available visualizations are ready.',
                )
                update_audio_progress(
                    'analysis_completed',
                    'complete',
                    'Descriptive audio analysis completed; no individual diagnosis or risk result was generated.',
                )
                status.update(label='Analysis completed',state='complete',expanded=True)
                mark_assessment_changed()
                st.rerun()
            except Exception as exc:
                logger.error(
                    "Audio assessment processing failed (%s).",
                    type(exc).__name__,
                )
                update_audio_progress(
                    'audio_analysis',
                    'error',
                    'Audio could not be processed. Check the file format and limits, then retry.',
                )
                st.session_state.audio_result=None
                st.session_state.voice_wav_bytes=None
                st.session_state.voice_visualization=None
                mark_assessment_changed()
                st.error(
                    f'Audio could not be processed. Check the format, duration, '
                    f'and recording quality, then retry.'
                )
        if not analysis_ran and st.session_state.get('audio_pipeline_events'):
            events=st.session_state.audio_pipeline_events
            failed=events[-1].get('state')=='error'
            with st.status(
                'Audio analysis failed' if failed else 'Audio analysis completed',
                state='error' if failed else 'complete',
                expanded=False,
            ) as completed_status:
                for event in events:
                    if event.get('state') in ('complete','skipped','error'):
                        completed_status.write(
                            f"{event['stage']} · {event['message']}"
                        )
        r=st.session_state.audio_result
        if r:
            q=r['quality']
            score=q.get('score')
            quality_label=q.get('label','Quality details unavailable')
            st.metric(
                'Recording quality',
                f"{score}/100 · {quality_label}" if score is not None else quality_label,
            )
            st.caption(q.get(
                'meaning',
                'Technical recording quality only; not a health or disease-risk score.',
            ))

            def audio_value(key,format_string):
                value=r.get(key)
                return format_string.format(value) if isinstance(value,(int,float,np.number)) else 'Unavailable'

            source_rate_key=(
                'source_sample_rate' if r.get('source_sample_rate') is not None
                else 'sample_rate'
            )
            source_metadata_available=(
                r.get('source_sample_rate') is not None
                and r.get('source_channels') is not None
            )
            metadata_cols=st.columns(4)
            metadata_cols[0].metric('Duration',f"{audio_value('duration_seconds','{:.1f}')} s")
            metadata_cols[1].metric(
                'Source sample rate' if source_metadata_available else 'Analysis sample rate',
                f"{audio_value(source_rate_key,'{:,.0f}')} Hz",
            )
            metadata_cols[2].metric(
                'Source channels',
                str(r.get('source_channels','Unavailable')),
            )
            metadata_cols[3].metric(
                'Near-silence frames',
                audio_value('silence_ratio','{:.1%}'),
            )
            feature_cols=st.columns(4)
            feature_cols[0].metric('RMS level',audio_value('rms','{:.4f}'))
            feature_cols[1].metric('Peak amplitude',audio_value('peak_amplitude','{:.3f}'))
            feature_cols[2].metric(
                'Median pitch estimate',
                f"{audio_value('pitch_median_hz','{:.1f}')} Hz"
                if r.get('pitch_median_hz') is not None else 'Unavailable',
            )
            feature_cols[3].metric(
                'RMS frame variability',
                audio_value('rms_variability','{:.4f}'),
            )
            if q.get('reasons'):
                st.warning('Review the recording quality notes before interpreting these descriptive measurements.')
                for reason in q['reasons']:
                    st.caption(f'· {reason}')
            visualization=st.session_state.get('voice_visualization')
            if visualization:
                st.subheader('Audio signal')
                if source_metadata_available:
                    st.caption(
                        f"Source: {audio_value(source_rate_key,'{:,.0f}')} Hz, "
                        f"{r.get('source_channels')} channel(s), "
                        f"{audio_value('source_duration_seconds','{:.1f}')} s. "
                        f"Analysis waveform: {audio_value('sample_rate','{:,.0f}')} Hz mono PCM WAV."
                    )
                else:
                    st.caption(
                        f"Original input metadata is unavailable for this resumed "
                        f"draft. The available analysis waveform is "
                        f"{audio_value('sample_rate','{:,.0f}')} Hz mono PCM WAV."
                    )
                if st.session_state.get('voice_wav_bytes'):
                    st.audio(
                        st.session_state.voice_wav_bytes,
                        format='audio/wav',
                        alt='Converted mono WAV used for descriptive audio analysis',
                    )
                waveform_col,spectrogram_col=st.columns(2)
                waveform_figure=go.Figure(go.Scatter(
                    x=visualization['waveform_time_seconds'],
                    y=visualization['waveform_amplitude'],
                    mode='lines',
                    line={'color':'#16858A','width':1},
                    hovertemplate='%{x:.2f} s<br>Amplitude %{y:.3f}<extra></extra>',
                ))
                waveform_figure.update_layout(
                    title='Waveform',
                    xaxis_title='Time (seconds)',
                    yaxis_title='Amplitude',
                )
                waveform_col.plotly_chart(
                    style_research_figure(waveform_figure),
                    width='stretch',
                    alt='Downsampled waveform of the converted audio over time',
                )
                spectrogram_figure=go.Figure(go.Heatmap(
                    x=visualization['spectrogram_time_seconds'],
                    y=visualization['spectrogram_frequency_hz'],
                    z=visualization['spectrogram_db'],
                    colorscale='Viridis',
                    colorbar={'title':'dB'},
                    hovertemplate='%{x:.2f} s<br>%{y:.0f} Hz<br>%{z:.1f} dB<extra></extra>',
                ))
                spectrogram_figure.update_layout(
                    title='Short-time magnitude spectrogram',
                    xaxis_title='Time (seconds)',
                    yaxis_title='Frequency (Hz)',
                )
                spectrogram_col.plotly_chart(
                    style_research_figure(spectrogram_figure),
                    width='stretch',
                    alt='Short-time audio spectrum by frequency and time, expressed in relative decibels',
                )
            if r.get('mfcc_means'):
                voice_figure=go.Figure(go.Bar(
                    x=list(range(1,len(r['mfcc_means'])+1)),
                    y=r['mfcc_means'],
                    marker_color='#16858A',
                    hovertemplate='Coefficient %{x}<br>Mean value %{y:.3f}<extra></extra>',
                ))
                voice_figure.update_layout(
                    title='Exploratory acoustic coefficients',
                    xaxis_title='Coefficient',
                    yaxis_title='Mean value',
                )
                st.plotly_chart(
                    style_research_figure(voice_figure),
                    width='stretch',
                    alt='Exploratory bar chart of descriptive acoustic coefficients',
                )
            st.caption(
                'This voice path does not transcribe speech, estimate speech rate, '
                'or produce an individual disease probability. Features are sensitive '
                'to language, accent, age, task, device, and environment.'
            )
            feature_explanations=[
                {
                    'Measurement':'Waveform',
                    'Recorded value':'Shown above',
                    'What it measures':'Amplitude samples over time in the converted mono recording.',
                    'Why examined':'Lets a reviewer inspect signal shape and obvious recording issues.',
                    'Limitations':'Amplitude is not calibrated loudness; the display is downsampled for readability.',
                },
                {
                    'Measurement':'Spectrogram',
                    'Recorded value':'Shown above · relative dB',
                    'What it measures':'Short-time magnitude by frequency and time.',
                    'Why examined':'Lets a reviewer inspect broad spectral structure in the recording.',
                    'Limitations':'Relative display only; no speech meaning or clinical interpretation is inferred.',
                },
                {
                    'Measurement':'Duration',
                    'Recorded value':f"{audio_value('duration_seconds','{:.2f}')} s",
                    'What it measures':'Elapsed time in the analyzed audio.',
                    'Why examined':'Provides context for the signal and task length.',
                    'Limitations':'Does not measure speaking rate or indicate health status.',
                },
                {
                    'Measurement':'Near-silence frame ratio',
                    'Recorded value':audio_value('silence_ratio','{:.1%}'),
                    'What it measures':'Share of 25 ms frames below the existing low-energy threshold.',
                    'Why examined':'Flags recordings dominated by near-silence.',
                    'Limitations':'Not a count or duration of speech pauses; affected by microphone gain and background noise.',
                },
                {
                    'Measurement':'RMS and peak amplitude',
                    'Recorded value':(
                        f"RMS {audio_value('rms','{:.4f}')}; "
                        f"peak {audio_value('peak_amplitude','{:.3f}')}"
                    ),
                    'What it measures':'Digital signal energy and maximum sample amplitude.',
                    'Why examined':'Helps review recording level and possible clipping.',
                    'Limitations':'Not calibrated loudness or vocal effort; depends on device and distance.',
                },
                {
                    'Measurement':'Clipping ratio',
                    'Recorded value':audio_value('clipping_ratio','{:.3%}'),
                    'What it measures':'Share of samples at or above the existing near-maximum threshold.',
                    'Why examined':'Flags possible digital clipping in the input.',
                    'Limitations':'A technical signal check only; it is affected by recording level.',
                },
                {
                    'Measurement':'Median pitch estimate',
                    'Recorded value':(
                        f"{audio_value('pitch_median_hz','{:.1f}')} Hz"
                        if r.get('pitch_median_hz') is not None else 'Unavailable'
                    ),
                    'What it measures':'Median estimated fundamental frequency in the accepted pitch range.',
                    'Why examined':'A descriptive acoustic characteristic of voiced audio.',
                    'Limitations':'May be unavailable or inaccurate for noisy, unvoiced, or mixed speech; not a validated individual marker.',
                },
                {
                    'Measurement':'MFCC summaries',
                    'Recorded value':(
                        f"{len(r['mfcc_means'])} means"
                        if r.get('mfcc_means') else 'Unavailable'
                    ),
                    'What it measures':'Compact summaries of short-time spectral envelope coefficients.',
                    'Why examined':'Describes broad acoustic shape for exploratory signal review.',
                    'Limitations':'Not semantic or clinical interpretations; affected by task, language, device, and environment.',
                },
                {
                    'Measurement':'Zero-crossing rate',
                    'Recorded value':audio_value('zero_crossing_rate','{:.4f}'),
                    'What it measures':'Average rate of waveform sign changes.',
                    'Why examined':'Basic descriptor of signal texture.',
                    'Limitations':'Sensitive to noise and recording setup; not specific to a condition.',
                },
                {
                    'Measurement':'Spectral centroid',
                    'Recorded value':f"{audio_value('spectral_centroid_hz','{:.1f}')} Hz",
                    'What it measures':'Average frequency-weighted center of the spectrum.',
                    'Why examined':'Summarizes spectral brightness for exploratory comparison.',
                    'Limitations':'Varies with phoneme, microphone, and environment; not a validated biomarker.',
                },
                {
                    'Measurement':'RMS frame variability',
                    'Recorded value':audio_value('rms_variability','{:.4f}'),
                    'What it measures':'Variation in short-time signal energy.',
                    'Why examined':'Describes amplitude changes over the recording.',
                    'Limitations':'Not pitch variability or a clinical measure; affected by speaking style and microphone distance.',
                },
                {
                    'Measurement':'Technical recording quality',
                    'Recorded value':(
                        f"{score}/100 · {quality_label}" if score is not None else quality_label
                    ),
                    'What it measures':'Existing checks for duration, level, clipping, sample rate, and near-silence.',
                    'Why examined':'Indicates whether the signal may be usable for descriptive analysis.',
                    'Limitations':'Quality score only; not model confidence, disease probability, or clinical uncertainty.',
                },
            ]
            with st.expander('Understanding your analysis'):
                st.dataframe(
                    pd.DataFrame(feature_explanations),
                    hide_index=True,
                    alt='Descriptions and limitations for each displayed audio measurement',
                )

        # New feature section starts here (properly indented with 8 spaces)
        st.divider()
        st.subheader('Experimental symptom-pattern analysis')
        st.write("Sustain a steady vowel such as 'ahhh' for at least 5 seconds, then run the research pattern comparison.")

        if is_demo_mode(user):
            st.caption('Demo Mode uses generated sample audio only; recording and file uploads are disabled.')
            pattern_rec=None
            pattern_upload=None
        else:
            pattern_rec = st.audio_input(
                'Record a sustained vowel for pattern analysis',
                key=f"pattern_rec{st.session_state.upload_generation}",
                on_change=clear_pattern_demo,
            )
            pattern_upload = st.file_uploader(
                'Or upload a WAV for pattern analysis',
                type=['wav'],
                key=f"pattern_wav{st.session_state.upload_generation}",
                on_change=clear_pattern_demo,
            )
        pattern_source = pattern_rec or pattern_upload or chosen
        pattern_demo_bytes = st.session_state.get('pattern_demo_audio_bytes')
        pattern_demo_active = st.session_state.get('pattern_demo_active', False)
        if pattern_demo_active:
            pattern_audio = pattern_demo_bytes
        elif pattern_source is not None:
            pattern_audio = pattern_source.getvalue()
        else:
            pattern_audio = None

        run_col, sample_col = st.columns(2)
        with run_col:
            run_pattern = st.button(
                'Run experimental pattern analysis',
                disabled=pattern_audio is None,
            )
        with sample_col:
            sample_label=(
                'Load synthetic demo vowel sample'
                if is_demo_mode(user)
                else 'Load Sample 6-Second Vowel Audio'
            )
            load_sample = st.button(sample_label)

        if load_sample:
            st.session_state['pattern_demo_audio_bytes'] = generate_sample_vowel_wav()
            st.session_state['pattern_demo_active'] = True
            pattern_audio = st.session_state['pattern_demo_audio_bytes']
            pattern_demo_active = True
            run_pattern = True

        if pattern_audio is None:
            if is_demo_mode(user):
                st.info('Load the synthetic sample above to try the optional research model pathway.')
            else:
                st.info('No audio recorded yet. Record a steady vowel for at least 5 seconds or upload a WAV file to run the experimental analysis.')
        elif pattern_demo_active:
            st.audio(pattern_audio, format='audio/wav', alt='Six-second synthetic harmonic vowel sample')

        if run_pattern:
            pattern_status=st.status(
                'Audio received for experimental pattern analysis',
                expanded=True,
            )
            pattern_status.write(
                'This optional route accepts only a sustained-vowel task and uses the existing research classifier.'
            )
            pattern_titles={
                'validation':'Audio validation',
                'conversion':'Model input preparation',
                'preprocessing':'Signal preprocessing',
                'feature_extraction':'Praat feature extraction',
                'model_inference':'Research model loading and inference',
                'results_preparation':'Results preparation',
            }

            def update_pattern_progress(stage,state,message):
                title=pattern_titles.get(stage,stage.replace('_',' ').capitalize())
                if state=='running':
                    pattern_status.update(
                        label=f'{title} in progress',
                        state='running',
                        expanded=True,
                    )
                elif state=='complete':
                    pattern_status.write(f'✓ {title} · {message}')
                    pattern_status.update(
                        label=f'{title} complete',
                        state='running',
                        expanded=True,
                    )

            try:
                from modules.praat_features import run_symptom_pattern_analysis
                from modules.audio_analysis import to_standard_wav

                src=pattern_audio
                if not pattern_demo_active:
                    src=to_standard_wav(
                        pattern_audio,
                        progress_callback=update_pattern_progress,
                    )[0]
                st.session_state.pattern_result=run_symptom_pattern_analysis(
                    src,
                    progress_callback=update_pattern_progress,
                )
                pattern_status.update(
                    label='Experimental pattern analysis completed',
                    state='complete',
                    expanded=True,
                )
                mark_assessment_changed()
            except ValueError as exc:
                logger.warning(
                    "Experimental voice pattern analysis was rejected (%s).",
                    type(exc).__name__,
                )
                st.session_state.pattern_result=None
                pattern_status.update(
                    label='Experimental pattern analysis could not run',
                    state='error',
                    expanded=True,
                )
                mark_assessment_changed()
                if 'Too little voiced audio' in str(exc):
                    st.info(
                        'Pattern analysis was not run because the recording did '
                        'not contain enough usable voiced audio.'
                    )
                else:
                    st.error(
                        'Pattern analysis could not process this audio. Check the '
                        'WAV format and recording limits, then retry.'
                    )
            except Exception as exc:
                logger.error(
                    "Experimental voice pattern analysis failed (%s).",
                    type(exc).__name__,
                )
                st.session_state.pattern_result=None
                pattern_status.update(
                    label='Experimental pattern analysis failed',
                    state='error',
                    expanded=True,
                )
                mark_assessment_changed()
                st.warning(
                    'Experimental pattern analysis is unavailable. Try again '
                    'later or continue with the other optional activities.'
                )

        pr = st.session_state.get('pattern_result')
        if pr:
            if pr['recording_quality']['score'] < 60:
                st.warning('Recording quality is low; the pattern output below is unreliable. Re-record in a quiet room.')
            st.metric('Research pattern result', pr['experimental_output']['pattern_label'])
            model_cols=st.columns(2)
            model_score=pr['experimental_output'].get('model_score')
            model_threshold=pr['experimental_output'].get('threshold')
            model_cols[0].metric(
                'Research model score · not confidence',
                f'{model_score:.3f}' if model_score is not None else 'Unavailable',
            )
            model_cols[1].metric(
                'Research decision threshold',
                f'{model_threshold:.3f}' if model_threshold is not None else 'Unavailable',
            )
            balanced_accuracy=pr['model_context'].get('subject_level_balanced_accuracy')
            ci95=pr['model_context'].get('ci95')
            if balanced_accuracy is not None and ci95:
                st.caption(
                    f"Model context: subject-level balanced accuracy "
                    f"{balanced_accuracy:.2f} (95% CI {ci95[0]:.2f}–{ci95[1]:.2f}) · "
                    'internal validation only; this is not person-level uncertainty.'
                )
            st.caption(
                'The score is an uncalibrated model output for a task-specific research '
                'comparison, not confidence, disease probability, or an individual diagnosis.'
            )
            with st.expander('Measurements extracted'):
                praat_rows=[]
                for feature,value in pr['compatible_features'].items():
                    what,why=PRAAT_FEATURE_EXPLANATIONS.get(
                        feature,
                        (
                            'Feature produced by the sustained-vowel extractor.',
                            'Input to the existing research classifier.',
                        ),
                    )
                    praat_rows.append({
                        'Feature':feature.replace('_',' ').capitalize(),
                        'Recorded value':(
                            f'{value:.5g}' if isinstance(value,(int,float)) else str(value)
                        ),
                        'What it measures':what,
                        'Why examined':why,
                        'Limitations':(
                            'Task-specific sustained-vowel estimate; sensitive to '
                            'phonation, microphone, noise, and Praat settings. '
                            'Not an individual clinical marker.'
                        ),
                    })
                st.dataframe(
                    pd.DataFrame(praat_rows),
                    hide_index=True,
                    alt='All extracted sustained-vowel features with measurement descriptions and limitations',
                )
            with st.expander('Understanding the research model output'):
                st.write(
                    'The existing classifier compares this task-specific feature '
                    'vector with research dataset patterns. It is not a diagnostic '
                    'model for general speech or an estimate of disease probability.'
                )
                st.dataframe(
                    pd.DataFrame([
                        {
                            'Output':'Research model score',
                            'Recorded value':(
                                f"{model_score:.3f}" if model_score is not None else 'Unavailable'
                            ),
                            'Meaning':'Classifier output used with its stored threshold.',
                            'Limitation':'Not calibrated confidence, clinical risk, or person-level uncertainty.',
                        },
                        {
                            'Output':'Research decision threshold',
                            'Recorded value':(
                                f"{model_threshold:.3f}" if model_threshold is not None else 'Unavailable'
                            ),
                            'Meaning':'Stored threshold used for the displayed pattern label.',
                            'Limitation':'A model decision setting, not a clinical cut-off.',
                        },
                        {
                            'Output':'Subject-level validation balanced accuracy',
                            'Recorded value':(
                                f'{balanced_accuracy:.3f}' if balanced_accuracy is not None else 'Unavailable'
                            ),
                            'Meaning':'Aggregate internal cross-validation metric.',
                            'Limitation':'Describes cohort-level validation, not this participant.',
                        },
                    ]),
                    hide_index=True,
                    alt='Explanation and limitations of the research model score, threshold, and validation metric',
                )
            st.info(pr['required_disclaimer'])
    with motion_tab:
        st.subheader('Resting motion'); st.write('Record a short resting sensor sample using the companion phone logger, then upload its CSV.')
        if is_demo_mode(user):
            st.info('Sensor uploads are disabled in Demo Mode. This keeps uploaded participant recordings out of the demo.')
        else:
            mf=st.file_uploader('Upload motion CSV',type=['csv'],key=f"motion{st.session_state.upload_generation}")
            if st.button('Analyze resting motion',disabled=mf is None):
                with st.spinner('Checking the motion file and extracting descriptive measurements…'):
                    try:
                        frame=read_sensor_csv(mf.getvalue())
                        st.session_state.motion_result=analyze_motion(frame)
                        mark_assessment_changed()
                        st.success('Motion features extracted.')
                        st.rerun()
                    except Exception as exc:
                        logger.error(
                            "Motion CSV processing failed (%s).",
                            type(exc).__name__,
                        )
                        st.session_state.motion_result=None
                        mark_assessment_changed()
                        st.error(
                            'Motion CSV could not be processed. Check that it is a '
                            'valid sensor file within the upload limits and retry.'
                        )
        r=st.session_state.motion_result
        if r:
            st.metric('Signal quality',f"{r['quality']['score']}/100"); a,b,c=st.columns(3); a.metric('Duration',f"{r['duration_seconds']:.1f}s"); b.metric('Sampling rate',f"{r['sample_rate_hz']:.1f}Hz"); c.metric('Dominant frequency',f"{r['dominant_frequency_hz']:.2f}Hz" if r['dominant_frequency_hz'] else 'Unavailable')
            if (
                isinstance(r.get('frequencies_hz'), list)
                and isinstance(r.get('spectrum'), list)
            ):
                motion_figure=go.Figure(go.Scatter(
                    x=r['frequencies_hz'],
                    y=r['spectrum'],
                    mode='lines',
                    line={'color':'#3279A7','width':2.5},
                    hovertemplate='%{x:.2f} Hz<br>Magnitude %{y:.3f}<extra></extra>',
                ))
                motion_figure.update_layout(
                    title='Exploratory motion spectrum',
                    xaxis_title='Frequency (Hz)',
                    yaxis_title='Magnitude',
                )
                st.plotly_chart(
                    style_research_figure(motion_figure),
                    width='stretch',
                    alt='Exploratory motion spectrum by frequency',
                )
            else:
                st.info(
                    'The detailed spectrum is not retained in the local resume '
                    'draft; the saved summary measurements are still available.'
                )
            st.caption('Frequency peaks are nonspecific and do not diagnose Parkinson’s disease.')
    with gait_tab:
        st.subheader('Gait assessment'); st.write('Only perform this if walking is safe for you. Use a level, clear path and stop if dizzy or unsteady.')
        if is_demo_mode(user):
            st.info('Sensor uploads are disabled in Demo Mode. This keeps uploaded participant recordings out of the demo.')
        else:
            gf=st.file_uploader('Upload gait CSV',type=['csv'],key=f"gait{st.session_state.upload_generation}")
            if st.button('Analyze gait',disabled=gf is None):
                with st.spinner('Checking the gait file and estimating descriptive cadence…'):
                    try:
                        frame=read_sensor_csv(gf.getvalue())
                        st.session_state.gait_result=analyze_gait(frame)
                        mark_assessment_changed()
                        st.success('Exploratory gait features extracted.')
                        st.rerun()
                    except Exception as exc:
                        logger.error(
                            "Gait CSV processing failed (%s).",
                            type(exc).__name__,
                        )
                        st.session_state.gait_result=None
                        mark_assessment_changed()
                        st.error(
                            'Gait CSV could not be processed. Check that it is a '
                            'valid sensor file within the upload limits and retry.'
                        )
        r=st.session_state.gait_result
        if r:
            a,b,c=st.columns(3); a.metric('Signal quality',f"{r['quality']['score']}/100"); b.metric('Candidate step events',r['candidate_step_events']); c.metric('Estimated cadence',f"{r['cadence_steps_per_min']:.1f}/min" if r['cadence_steps_per_min'] is not None else 'Unavailable'); st.caption(r['method_note'])
    with clock_tab:
        st.subheader('Clock-drawing activity')
        st.write('Draw a clock face, add the numbers, and set the hands to 10 past 11.')
        sample_drawing=st.session_state.get('clock_sample_drawing')
        canvas=st_canvas(
            fill_color='rgba(13,148,136,0.12)',
            stroke_width=3,
            stroke_color='#0d192b',
            background_color='#ffffff',
            height=340,
            width=520,
            drawing_mode='freedraw',
            initial_drawing=sample_drawing,
            key=(
                f"clock_canvas_{st.session_state.upload_generation}_"
                f"{st.session_state.get('clock_canvas_generation', 0)}"
            ),
        )
        drawing_json=getattr(canvas,'json_data',None)
        if not isinstance(drawing_json,dict) and sample_drawing:
            drawing_json=sample_drawing
        has_strokes=isinstance(drawing_json,dict) and bool(drawing_json.get('objects'))
        clear_col,sample_col,analyze_col=st.columns(3)
        clear_col.button(
            'Clear Canvas',
            on_click=clear_clock_canvas,
            width='stretch',
        )
        sample_col.button(
            'Load Sample Clock Drawing',
            on_click=load_sample_clock_drawing,
            width='stretch',
        )
        st.caption(
            'This task records coarse drawing geometry only; it does not calculate '
            'Alzheimer’s risk or provide a diagnosis. The browser canvas has no '
            'calibrated pressure or reliable pointer timestamps.'
        )
        analyze_drawing=analyze_col.button(
            'Analyze drawing',
            disabled=not has_strokes,
            key='analyze_clock_drawing',
            type='primary',
            width='stretch',
        )
        if analyze_drawing:
            try:
                drawing_features=extract_clock_drawing_features(drawing_json)
                darwin_status=get_alzheimers_status()
                selected_darwin_features=(
                    (darwin_status.get('metadata') or {})
                    .get('feature_selection',{})
                    .get('selected_features',[])
                )
                darwin_compatibility=assess_darwin_feature_compatibility(
                    drawing_features,
                    selected_darwin_features,
                )
                st.session_state.clock_result={
                    'completed':True,
                    'task':'Draw a clock showing 11:10',
                    'saved_at':datetime.now(timezone.utc).isoformat(),
                    'scoring':'No clinical or kinematic similarity score is produced.',
                    'features':drawing_features,
                    'darwin_model_compatibility':darwin_compatibility,
                }
                mark_assessment_changed()
                st.success('Drawing geometry recorded for this session.')
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if st.session_state.clock_result:
            clock_features=st.session_state.clock_result.get('features')
            if clock_features:
                with st.container(border=True):
                    st.subheader('Geometric & Kinematic Capture Card')
                    metric_cols=st.columns(4)
                    metric_cols[0].metric('Stroke count',clock_features['stroke_count'])
                    metric_cols[1].metric(
                        'Total path length',
                        f"{clock_features['approx_path_length_px']:.1f} px",
                    )
                    metric_cols[2].metric(
                        'Canvas area used',
                        f"{clock_features['bounding_box_area_fraction']:.1%}",
                    )
                    duration=clock_features['drawing_duration_seconds']
                    metric_cols[3].metric(
                        'Drawing duration',
                        f'{duration:.1f} s' if duration is not None else 'Unavailable',
                    )
                compatibility=st.session_state.clock_result.get('darwin_model_compatibility') or {}
                st.caption(compatibility.get('message','No compatible DARWIN feature schema is available.'))
                st.caption(clock_features['timing_limitation'])
            else:
                st.json(st.session_state.clock_result)
        darwin_status=get_alzheimers_status()
        if darwin_status.get('available'):
            st.caption(
                'DARWIN dataset note: the model is available for dataset-level research '
                'evaluation only. Its 450 tablet features do not match this single-task '
                'browser canvas, so no user-level similarity or prediction is produced.'
            )
        else:
            st.caption(
                'DARWIN dataset note: '
                + darwin_status.get('message','DARWIN research model is not installed.')
            )
        with st.expander('DARWIN dataset-level research example'):
            if st.button(
                'Load Sample DARWIN Test Profile',
                type='secondary',
                disabled=not darwin_status.get('available'),
                key='load_darwin_sample_profile',
            ):
                try:
                    st.session_state.darwin_sample_profile=load_darwin_sample_profile()
                    st.rerun()
                except (RuntimeError, KeyError) as exc:
                    st.error(f'Could not load the DARWIN cohort example: {exc}')
            sample_profile=st.session_state.get('darwin_sample_profile')
            if sample_profile:
                st.info(sample_profile['notice'])
                st.dataframe(
                    pd.DataFrame([
                        {'Feature':humanize_feature_code(name),'Cohort median':value}
                        for name,value in sample_profile['features'].items()
                    ]),
                    hide_index=True,
                    alt='Dataset-wide median values for selected DARWIN research features',
                )
    with reaction_tab:
        st.subheader('Visual reaction-time activity'); st.write('Press Start, wait for the green signal, then tap as quickly as you comfortably can. Complete five trials.')
        if st.button('Start / reset reaction trial',type='primary'):
            st.session_state.reaction_started=time.perf_counter(); st.session_state.reaction_ready=False; st.session_state.reaction_delay=random.uniform(1.5,3.5)
        if st.session_state.reaction_started is not None:
            elapsed=time.perf_counter()-st.session_state.reaction_started
            if elapsed>=st.session_state.get('reaction_delay',2): st.session_state.reaction_ready=True
            if st.session_state.reaction_ready:
                st.success('Signal shown — tap now')
                if st.button('I saw it — tap!',key=f"react{len(st.session_state.reaction_times)}"):
                    ms=(time.perf_counter()-st.session_state.reaction_started)*1000-st.session_state.get('reaction_delay',2)*1000
                    st.session_state.reaction_times.append(max(0,round(ms,1))); mark_assessment_changed(); st.session_state.reaction_started=None; st.session_state.reaction_ready=False; st.rerun()
            else: st.info('Wait for the green signal…'); time.sleep(.25); st.rerun()
        if st.session_state.reaction_times:
            summary=reaction_summary(st.session_state.reaction_times); a,b,c=st.columns(3); a.metric('Trials',summary['trials']); b.metric('Median response',f"{summary['median_ms']} ms"); c.metric('Response variability',f"{summary['variability_ms']} ms"); st.caption('Timing is affected by browser and device latency; this is not a clinical cognitive test.')
            if st.button('Clear reaction trials'): st.session_state.reaction_times=[]; mark_assessment_changed(); st.rerun()
    with memory_tab:
        st.subheader('Interactive memory assessment')
        render_memory_check(st.session_state,mark_assessment_changed)
    with alz_tab:
        st.subheader("Alzheimer's speech features — research only")
        if is_demo_mode(user):
            st.info(
                'Transcript and matching-audio inputs are disabled in Demo Mode '
                'to avoid collecting personal narratives. No speech model is run.'
            )
        else:
            st.write('Describe your morning or a recent event in a few sentences (at least 10 words), or paste what you said in the voice task above.')
            st.caption('No trained model exists for this pathway yet: you receive measurements only, never a score. Features require a transcript; audio adds timing features.')
            transcript = st.text_area('Transcript', height=120, key=f"alz_transcript_{st.session_state.upload_generation}",
                placeholder='For example: I woke up at seven, made coffee, and walked the dog around the block before breakfast...')
            transcript_word_count=len(re.findall(r"[a-z'\-]+",transcript.lower()))
            if transcript_word_count < 10:
                st.info(f'Enter at least 10 words to enable extraction ({transcript_word_count}/10 entered).')
            alz_audio = st.file_uploader('Optional: matching audio file (any format — auto-converted to WAV)', type=['wav','mp3','m4a','ogg','flac'], key=f"alz_audio_{st.session_state.upload_generation}")
            if st.button('Extract speech features', disabled=transcript_word_count < 10):
                with st.spinner('Extracting descriptive transcript features…'):
                    try:
                        from modules.alzheimers_model import run_alzheimers_feature_report
                        from modules.audio_analysis import to_standard_wav
                        if alz_audio is not None:
                            wb, _ = to_standard_wav(alz_audio.getvalue())
                            import tempfile
                            with tempfile.TemporaryDirectory(prefix='neuronexus-speech-') as tmp_dir:
                                audio_path=Path(tmp_dir)/'matching.wav'
                                audio_path.write_bytes(wb)
                                st.session_state.alz_result = run_alzheimers_feature_report(transcript, str(audio_path))
                        else:
                            st.session_state.alz_result = run_alzheimers_feature_report(transcript)
                        mark_assessment_changed()
                        st.rerun()
                    except Exception as exc:
                        logger.error(
                            "Speech-feature extraction failed (%s).",
                            type(exc).__name__,
                        )
                        st.session_state.alz_result = None
                        mark_assessment_changed()
                        st.error(
                            'Speech features could not be extracted. Review the '
                            'input format and length, then retry.'
                        )
        ar = st.session_state.get('alz_result')
        if ar:
            feats = ar['features']
            c = st.columns(4)
            for col, (label, key, fmt) in zip(c, [('Words', 'n_words', '{:.0f}'), ('Type-token ratio', 'type_token_ratio', '{:.3f}'),
                                                  ('Pronoun ratio', 'pronoun_ratio', '{:.3f}'), ('Idea density', 'idea_density', '{:.3f}')]):
                col.metric(label, fmt.format(feats[key]))
            if ar['audio_used']:
                c2 = st.columns(3)
                for col, (label, key, fmt) in zip(c2, [('Duration', 'duration_seconds', '{:.1f} s'),
                                                       ('Words per minute', 'words_per_minute', '{:.1f}'), ('Pause ratio', 'pause_ratio', '{:.1%}')]):
                    if key in feats: col.metric(label, fmt.format(feats[key]))
            with st.expander('All extracted features'):
                st.json({k: (round(v, 5) if isinstance(v, float) else v) for k, v in feats.items()})
            st.info(ar['note'])

    with report_tab:
        st.subheader('Session summary'); report=build_report(audio_result=st.session_state.audio_result,motion_result=st.session_state.motion_result,gait_result=st.session_state.gait_result,mode=st.session_state.get('screening_mode','Individual'))
        saved_report_notice=st.session_state.pop('report_saved_notice',None)
        if saved_report_notice:
            st.success(saved_report_notice)
        if is_demo_mode(user):
            report['demo_mode']=True
            report['demo_notice']=DEMO_MODE_NOTICE
        report['cognitive_tasks']={
            'clock_drawing':st.session_state.clock_result,
            'reaction_time':reaction_summary(st.session_state.reaction_times),
            'memory_check':st.session_state.get('memory_result'),
        }
        if st.session_state.clock_result and st.session_state.clock_result.get('features'):
            report['clock_drawing_measurements']=st.session_state.clock_result['features']
        _ar=st.session_state.get('alz_result')
        if _ar: report['alzheimers_speech_features']=_ar['features']
        pattern_result=st.session_state.get('pattern_result')
        if pattern_result:
            report['voice_pattern_features']=pattern_result.get('compatible_features') or {}
            report['model_outputs']={
                **(report.get('model_outputs') or {}),
                'sustained_vowel_research_pattern':{
                    'experimental_output':pattern_result.get('experimental_output') or {},
                    'model_context':pattern_result.get('model_context') or {},
                    'required_disclaimer':pattern_result.get('required_disclaimer'),
                },
            }
        if st.session_state.get('darwin_sample_profile'):
            report['darwin_sample_profile']=st.session_state.darwin_sample_profile
        if st.session_state.get('consent_granted'):
            report['consent_flags']={
                'assessment':True,
                'optional_profile_context':bool(st.session_state.get('baseline_context')),
                'caregiver_observations':bool(st.session_state.get('caregiver_notes_consent')),
            }
            if st.session_state.get('baseline_context'):
                report['participant_context']=st.session_state.baseline_context
            if st.session_state.get('caregiver_notes_consent'):
                notes=st.session_state.get('caregiver_observation_notes') or []
                if notes:
                    report['caregiver_observations']=notes
            report['assisted_session_mode']=bool(st.session_state.get('assisted_session_mode'))
        report_session_id=st.session_state.get('last_saved_report_session_id')
        report_created_at=st.session_state.get('last_saved_report_created_at')
        if report_session_id is not None:
            report['session_id']=report_session_id
            if report_created_at:
                report['created_utc']=report_created_at
        completed_modules=completed_module_explanations(report)
        if not completed_modules:
            st.warning(
                'This session has no completed assessments. Complete at least one '
                'assessment before downloading or saving a screening report.'
            )
        st.info('This report contains measurements and task completion information, not a diagnosis or disease-risk estimate.')
        if is_demo_mode(user):
            st.warning(
                'This report belongs only to the current Demo Mode session. Saving '
                'adds it to the in-memory demo timeline; it is never written to an account.',
                title='Demo report · not a patient record',
            )
        st.json(report)
        report_history=sessions_for_user(user,st.session_state)
        report_wellness=saved_wellness_report_context(user)
        report_registry=get_registry()
        report_filename=(
            f'neuronexus_session_{report_session_id}.pdf'
            if report_session_id is not None
            else 'neuronexus_screening_report_unsaved.pdf'
        )
        try:
            pdf_data=make_pdf(
                user['name'],
                report,
                report_registry,
                session_id=report_session_id,
                session_created_at=report_created_at,
                history=report_history,
                wellness_plan=report_wellness,
            )
        except Exception as exc:
            logger.error(
                "Screening PDF generation failed (%s).",
                type(exc).__name__,
            )
            st.error(
                'The PDF report could not be generated. You can still download '
                'the structured JSON report or try again later.'
            )
        else:
            st.download_button(
                'Download PDF report',
                pdf_data,
                file_name=report_filename,
                mime='application/pdf',
                disabled=not completed_modules,
            )
        st.caption(
            'The PDF is print-ready. Open it in your browser or PDF reader to print.'
        )
        st.download_button(
            'Download JSON report',
            json.dumps(report,indent=2,allow_nan=False),
            file_name='neuronexus_report.json',
            mime='application/json',
            disabled=not completed_modules,
        )
        if is_demo_mode(user):
            if st.button(
                'Save report to demo timeline',
                type='primary',
                disabled=not completed_modules or report_session_id is not None,
            ):
                saved_id=save_demo_report(st.session_state,report)
                st.session_state['last_saved_report_session_id']=saved_id
                demo_session=next(
                    row for row in st.session_state['demo_sessions']
                    if row['id']==saved_id
                )
                st.session_state['last_saved_report_created_at']=demo_session['created_at']
                st.session_state.draft_finalized=True
                st.session_state['report_saved_notice']=(
                    f'Demo report saved to this session only. Session ID: {saved_id}.'
                )
                st.rerun()
        elif st.session_state.get('local_history_enabled'):
            if st.button(
                'Save session to my dashboard',
                type='primary',
                disabled=not completed_modules or report_session_id is not None,
            ):
                saved_id=save_session(
                    user['id'],
                    report,
                    created_at=report['created_utc'],
                )
                st.session_state['last_saved_report_session_id']=saved_id
                st.session_state['last_saved_report_created_at']=report['created_utc']
                delete_draft(user['id'])
                st.session_state.draft_finalized=True
                st.session_state['report_saved_notice']=(
                    f'Finalized report saved to your account. Session ID: {saved_id}.'
                )
                st.rerun()
        else:
            st.button(
                'Save session to my dashboard',
                type='primary',
                disabled=True,
            )
            st.caption('Enable saved session history in My profile to save this finalized report.')
    st.divider()
    if st.button('Clear current screening session'):
        reset_session(st.session_state, reset_consent=False); st.session_state.reaction_started=None; st.session_state.reaction_ready=False; st.rerun()

st.divider(); st.caption('NeuroNexus 2.0 prototype · Not a medical device · Experimental measurements only · Seek a qualified clinician for health concerns.')
