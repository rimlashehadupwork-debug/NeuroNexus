import streamlit as st
import plotly.graph_objects as go

from modules.audio_analysis import analyze_audio_bytes
from modules.config import (
    APP_NAME,
    APP_SUBTITLE,
    APP_VERSION,
    DIAGNOSTIC_DISCLAIMER,
    INTERFACE_LANGUAGES,
    SUPPORTED_MODES,
)
from modules.motion_analysis import analyze_motion
from modules.privacy import CONSENT_TEXT, PRIVACY_NOTICE
from modules.session import initialize_session, reset_session

st.set_page_config(page_title=APP_NAME, page_icon="🧠", layout="wide")
initialize_session(st.session_state)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 1.8rem; padding-bottom: 3rem;}
    .hero {padding: 1.4rem 1.6rem; border: 1px solid rgba(120,140,160,.25);
           border-radius: 18px; background: linear-gradient(120deg, rgba(20,184,166,.12), rgba(99,102,241,.08));}
    .small-muted {color: #8492a6; font-size: .92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Session setup")
    user_mode = st.radio("I am using NeuroNexus as", SUPPORTED_MODES)
    language = st.selectbox("Preferred interface language", INTERFACE_LANGUAGES)
    st.divider()
    st.caption("Privacy-conscious prototype")
    st.write("Uploads are analyzed for this session; this prototype does not intentionally save raw inputs to project files.")
    with st.expander("Read the data-handling notice"):
        st.write(PRIVACY_NOTICE)
    st.caption(f"Version {APP_VERSION} · {language}")
    if st.button("Clear session and start over", use_container_width=True, type="secondary"):
        reset_session(st.session_state)
        st.rerun()

st.markdown('<div class="hero">', unsafe_allow_html=True)
st.title("🧠 NeuroNexus")
st.write(APP_SUBTITLE)
st.markdown('<p class="small-muted">Voice • Resting motion • Gait roadmap • Signal Confidence</p>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.warning(DIAGNOSTIC_DISCLAIMER)

if user_mode == "Community health worker":
    st.subheader("Guided screening-session workspace")
    st.write("Use this workflow only with the participant's informed, voluntary consent. Explain the limitations before collecting any data.")
else:
    st.subheader("Your research session")
    st.write("Explore optional signal-analysis modules. You can clear the session at any time.")

consent_key = f"consent_{st.session_state.upload_generation}"
consent = st.checkbox(CONSENT_TEXT, key=consent_key)
if not consent:
    st.info("Consent is required before any sample can be analyzed. You can explore the interface without uploading data.")

voice_tab, tremor_tab, gait_tab, report_tab = st.tabs(
    ["🎙️ Voice", "📳 Resting motion", "🚶 Gait", "📋 Session report"]
)

with voice_tab:
    st.header("Acoustic analytics")
    st.write("Upload a WAV recording for this milestone. Keep it short and avoid names or other identifying information.")
    audio_file = st.file_uploader(
        "Choose a WAV file", type=["wav"], key=f"audio_upload_{st.session_state.upload_generation}"
    )
    st.caption("Current input: WAV upload. Browser-based microphone recording is planned for a later milestone.")
    if st.button("Analyze voice sample", disabled=not (consent and audio_file), key="analyze_audio"):
        try:
            st.session_state.audio_result = analyze_audio_bytes(audio_file.getvalue())
            st.success("Audio analysis completed.")
        except Exception as exc:
            st.session_state.audio_result = None
            st.error(f"Could not analyze this audio: {exc}")
    if st.session_state.audio_result:
        result = st.session_state.audio_result
        q = result["quality"]
        st.subheader("Signal confidence")
        if q["status"] == "usable":
            st.success(q["message"])
        else:
            st.warning(q["message"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Duration", f'{result["duration_seconds"]:.1f} s')
        c2.metric("Sample rate", f'{result["sample_rate"]:,} Hz')
        c3.metric("RMS level", f'{result["rms"]:.4f}')
        st.subheader("Measured acoustic features")
        cols = st.columns(3)
        cols[0].metric("MFCC feature count", str(len(result["mfcc_means"])))
        cols[1].metric("Pitch estimate", f'{result["pitch_median_hz"]:.1f} Hz' if result["pitch_median_hz"] is not None else "Unavailable")
        cols[2].metric("Zero-crossing rate", f'{result["zero_crossing_rate"]:.3f}')
        with st.expander("MFCC mean values"):
            st.write([round(x, 4) for x in result["mfcc_means"]])
        st.caption("These are signal measurements, not evidence that a person has or does not have a neurological condition.")

with tremor_tab:
    st.header("Resting motion analytics")
    st.write("Upload a CSV exported from a sensor logger. The file must contain numeric acceleration columns.")
    st.caption("Expected columns: timestamp (seconds, optional), ax, ay, az. Synthetic data is suitable for testing.")
    motion_file = st.file_uploader(
        "Choose motion CSV", type=["csv"], key=f"motion_upload_{st.session_state.upload_generation}"
    )
    if st.button("Analyze motion sample", disabled=not (consent and motion_file), key="analyze_motion"):
        try:
            import pandas as pd
            df = pd.read_csv(motion_file)
            st.session_state.motion_result = analyze_motion(df)
            st.success("Motion analysis completed.")
        except Exception as exc:
            st.session_state.motion_result = None
            st.error(f"Could not analyze this motion data: {exc}")
    if st.session_state.motion_result:
        result = st.session_state.motion_result
        if result["quality"]["status"] == "usable":
            st.success(result["quality"]["message"])
        else:
            st.warning(result["quality"]["message"])
        a, b, c = st.columns(3)
        a.metric("Samples", str(result["sample_count"]))
        b.metric("Estimated sample rate", f'{result["sample_rate_hz"]:.1f} Hz')
        c.metric("Dominant frequency", f'{result["dominant_frequency_hz"]:.2f} Hz' if result["dominant_frequency_hz"] is not None else "Unavailable")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result["frequencies_hz"], y=result["spectrum"], mode="lines", name="Acceleration spectrum"))
        fig.update_layout(title="Exploratory frequency spectrum", xaxis_title="Frequency (Hz)", yaxis_title="Relative amplitude", height=360)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("A frequency peak is not specific to Parkinson's disease and must not be interpreted as a diagnosis.")

with gait_tab:
    st.header("Gait feature roadmap")
    st.info("Gait analysis will be enabled after sensor capture and step-event detection are implemented.")
    st.write("Planned features: cadence, step-time variability, turn duration (if measurable), and data-quality checks.")
    st.write("Do not perform a walking task if you feel unsteady or unsafe.")

with report_tab:
    st.header("Session report")
    st.write("This report summarizes measurements held in the current app session.")
    if st.session_state.audio_result is None and st.session_state.motion_result is None:
        st.info("No analyses completed yet.")
    else:
        if st.session_state.audio_result:
            st.markdown("**Voice module**")
            st.json({
                "quality": st.session_state.audio_result["quality"],
                "duration_seconds": st.session_state.audio_result["duration_seconds"],
                "sample_rate": st.session_state.audio_result["sample_rate"],
                "pitch_median_hz": st.session_state.audio_result["pitch_median_hz"],
            })
        if st.session_state.motion_result:
            st.markdown("**Motion module**")
            st.json({
                "quality": st.session_state.motion_result["quality"],
                "sample_count": st.session_state.motion_result["sample_count"],
                "sample_rate_hz": st.session_state.motion_result["sample_rate_hz"],
                "dominant_frequency_hz": st.session_state.motion_result["dominant_frequency_hz"],
            })
    st.caption("No disease-risk score is calculated in this prototype.")
