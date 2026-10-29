import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.audio_analysis import analyze_audio_bytes
from modules.config import APP_NAME, APP_SUBTITLE, APP_VERSION, CONSENT_TEXT, DIAGNOSTIC_DISCLAIMER, PRIVACY_NOTICE, SUPPORTED_MODES
from modules.motion_analysis import analyze_motion, analyze_gait
from modules.session import initialize_session, reset_session
from modules.reporting import report_json

st.set_page_config(page_title=APP_NAME, page_icon="🧠", layout="wide")
initialize_session(st.session_state)

st.markdown("""
<style>
.block-container {max-width: 1240px; padding-top: 1.5rem; padding-bottom: 3rem;}
.hero {padding: 1.5rem 1.7rem; border: 1px solid rgba(120,140,160,.28); border-radius: 20px;
background: linear-gradient(120deg, rgba(20,184,166,.13), rgba(99,102,241,.10));}
.muted {color: #8492a6; font-size: .92rem;}
div[data-testid="stMetric"] {border: 1px solid rgba(120,140,160,.22); padding: .8rem; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Session setup")
    mode = st.radio("Workflow", SUPPORTED_MODES, key="workflow_mode")
    st.caption("Privacy-conscious research prototype")
    with st.expander("Data-handling notice"):
        st.markdown(PRIVACY_NOTICE)
    st.caption(f"Version {APP_VERSION}")
    if st.button("Clear session and start over", use_container_width=True):
        reset_session(st.session_state)
        st.rerun()

st.markdown('<div class="hero">', unsafe_allow_html=True)
st.title("🧠 NeuroNexus")
st.write(APP_SUBTITLE)
st.markdown('<p class="muted">Voice • Resting motion • Gait • Signal Confidence Engine</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.warning(DIAGNOSTIC_DISCLAIMER)

if mode == "Community health worker":
    st.subheader("Guided participant session")
    st.write("Explain the task and limitations in plain language. Participation must be voluntary; do not pressure anyone to continue.")
else:
    st.subheader("Your research session")
    st.write("Explore optional signal-analysis modules. Only use data you have permission to analyze.")

with st.expander("Before you begin: consent and safety", expanded=not st.session_state.consent_granted):
    st.markdown(PRIVACY_NOTICE)
    st.checkbox(CONSENT_TEXT, key="consent_granted")
if not st.session_state.consent_granted:
    st.info("Consent is required before analysis. You may explore the interface, but analysis actions are disabled.")

voice_tab, motion_tab, gait_tab, report_tab = st.tabs(["🎙️ Voice", "📳 Resting motion", "🚶 Gait", "📋 Session report"])

with voice_tab:
    st.header("Voice / acoustic features")
    st.write("Record a short sample in the browser or upload a WAV. A neutral sentence or sustained vowel can be used consistently across repeat sessions.")
    st.caption("Avoid names, addresses, and other identifying speech. Browser microphone access depends on browser support and a secure context.")
    recorded = st.audio_input("Record a voice sample", key=f"audio_record_{st.session_state.upload_generation}")
    audio_file = st.file_uploader("Or upload a WAV file", type=["wav"], key=f"audio_upload_{st.session_state.upload_generation}")
    selected_audio = recorded if recorded is not None else audio_file
    if st.button("Analyze voice", disabled=not (st.session_state.consent_granted and selected_audio is not None), key="run_voice"):
        try:
            st.session_state.audio_result = analyze_audio_bytes(selected_audio.getvalue())
            st.success("Audio features extracted.")
        except Exception as exc:
            st.session_state.audio_result = None
            st.error(str(exc))
    result = st.session_state.audio_result
    if result:
        q = result["quality"]
        (st.success if q["status"] == "usable" else st.warning)(f'{q["label"]} — technical quality score {q["score"]}/100')
        if q["reasons"]:
            for reason in q["reasons"]: st.write(f"• {reason}")
        a,b,c,d = st.columns(4)
        a.metric("Duration", f'{result["duration_seconds"]:.1f} s')
        b.metric("Sample rate", f'{result["sample_rate"]:,} Hz')
        c.metric("RMS level", f'{result["rms"]:.4f}')
        d.metric("Near-silence", f'{result["silence_ratio"]*100:.1f}%')
        a,b,c,d = st.columns(4)
        a.metric("Median pitch estimate", f'{result["pitch_median_hz"]:.1f} Hz' if result["pitch_median_hz"] is not None else "Unavailable")
        b.metric("Zero-crossing rate", f'{result["zero_crossing_rate"]:.3f}')
        c.metric("Spectral centroid", f'{result["spectral_centroid_hz"]:.0f} Hz')
        d.metric("RMS variability", f'{result["rms_variability"]:.4f}')
        st.plotly_chart(go.Figure(go.Bar(x=list(range(1,14)), y=result["mfcc_means"])).update_layout(title="Mean MFCC coefficients (exploratory)", xaxis_title="Coefficient", yaxis_title="Mean value"), use_container_width=True)
        st.caption("MFCC, pitch, and other acoustic measurements vary with language, accent, age, task, device, and environment. They are not diagnostic markers by themselves.")

with motion_tab:
    st.header("Resting motion / tremor-oriented exploration")
    st.write("Use a CSV from the companion sensor logger or your own authorized sensor export.")
    st.caption("Expected columns: timestamp (seconds, optional), ax, ay, az. Without timestamps the prototype assumes 50 Hz.")
    motion_file = st.file_uploader("Upload resting-motion CSV", type=["csv"], key=f"motion_upload_{st.session_state.upload_generation}")
    if st.button("Analyze resting motion", disabled=not (st.session_state.consent_granted and motion_file is not None), key="run_motion"):
        try:
            st.session_state.motion_result = analyze_motion(pd.read_csv(motion_file))
            st.success("Motion features extracted.")
        except Exception as exc:
            st.session_state.motion_result = None
            st.error(str(exc))
    result = st.session_state.motion_result
    if result:
        q = result["quality"]
        (st.success if q["status"] == "usable" else st.warning)(f'{q["label"]} — technical quality score {q["score"]}/100')
        for reason in q["reasons"]: st.write(f"• {reason}")
        a,b,c,d = st.columns(4)
        a.metric("Duration", f'{result["duration_seconds"]:.1f} s')
        b.metric("Samples", f'{result["sample_count"]:,}')
        c.metric("Sampling rate", f'{result["sample_rate_hz"]:.1f} Hz')
        d.metric("Dominant frequency", f'{result["dominant_frequency_hz"]:.2f} Hz' if result["dominant_frequency_hz"] is not None else "Unavailable")
        fig = go.Figure(go.Scatter(x=result["frequencies_hz"], y=result["spectrum"], mode="lines", name="Spectrum"))
        fig.update_layout(title="Exploratory frequency spectrum", xaxis_title="Frequency (Hz)", yaxis_title="Relative amplitude", height=360)
        st.plotly_chart(fig, use_container_width=True)
        if result["assumed_sample_rate"]:
            st.info("No timestamp column found; the 50 Hz sampling rate was assumed. This can materially affect frequency estimates.")
        st.caption("Frequency peaks are nonspecific and must not be used to diagnose or estimate Parkinson's disease.")

with gait_tab:
    st.header("Gait feature exploration")
    st.write("Record a short, safe walk using the phone sensor logger; upload its CSV here. Use a clear, level path and stop if you feel unsafe.")
    st.caption("This prototype uses an unvalidated acceleration-peak heuristic. Cadence and interval variability are exploratory, not clinical gait measurements.")
    gait_file = st.file_uploader("Upload gait CSV", type=["csv"], key=f"gait_upload_{st.session_state.upload_generation}")
    if st.button("Analyze gait sample", disabled=not (st.session_state.consent_granted and gait_file is not None), key="run_gait"):
        try:
            st.session_state.gait_result = analyze_gait(pd.read_csv(gait_file))
            st.success("Exploratory gait features extracted.")
        except Exception as exc:
            st.session_state.gait_result = None
            st.error(str(exc))
    result = st.session_state.gait_result
    if result:
        q = result["quality"]
        (st.success if q["status"] == "usable" else st.warning)(f'{q["label"]} — technical quality score {q["score"]}/100')
        for reason in q["reasons"]: st.write(f"• {reason}")
        a,b,c,d = st.columns(4)
        a.metric("Duration", f'{result["duration_seconds"]:.1f} s')
        b.metric("Sample rate", f'{result["sample_rate_hz"]:.1f} Hz')
        c.metric("Candidate step events", str(result["candidate_step_events"]))
        d.metric("Estimated cadence", f'{result["cadence_steps_per_min"]:.1f} steps/min' if result["cadence_steps_per_min"] is not None else "Unavailable")
        v = result["step_interval_variability_percent"]
        st.metric("Candidate interval variability", f'{v:.1f}%' if v is not None else "Unavailable")
        st.caption(result["method_note"])
        if result["assumed_sample_rate"]:
            st.info("No timestamps found; 50 Hz was assumed. Record timestamp values in seconds for better traceability.")

with report_tab:
    st.header("Session report")
    st.write("Download a compact JSON report containing derived features and technical quality notes. Raw recordings and raw sensor series are excluded.")
    has_results = any(st.session_state.get(k) is not None for k in ("audio_result", "motion_result", "gait_result"))
    if not has_results:
        st.info("Complete an analysis to populate the report.")
    else:
        report = report_json(audio_result=st.session_state.audio_result, motion_result=st.session_state.motion_result, gait_result=st.session_state.gait_result, mode=mode)
        st.code(report, language="json")
        st.download_button("Download session report (JSON)", data=report, file_name="neuronexus_session_report.json", mime="application/json")
        st.caption("The report records the workflow mode and report-generation time, but does not include a participant name or ID.")
    st.warning("This is a technical summary, not a clinical interpretation. Do not use it to make care decisions.")

st.divider()
st.caption("NeuroNexus 1.0.0 · Experimental research software · Review data governance and validation before any real-world study.")
