# NeuroNexus — Digital Biomarker Research Prototype

🚀 **Live Application:** [neuronexus18.streamlit.app](https://streamlit.app)  
🏆 **Built for:** Future Health Digital Innovation Challenge (Co-created by Microsoft, Inception, and MBZUAI)

---

## Overview
**NeuroNexus** is a browser-based multimodal digital-biomarker research prototype requiring zero specialized hardware. It offers voice, speech, movement, gait, reaction-time, clock-drawing, and an optional timed five-word recall task; descriptive session summaries; and a downloadable **Clinician Summary PDF**.

> **Medical & Research Disclaimer:** NeuroNexus is an exploratory research and triage prototype. It does **not** diagnose, rule out, or estimate the clinical probability of Alzheimer’s disease, Parkinson’s disease, or any other neurological condition.

---

## Key Capabilities
* **Guided exploratory activities:** Select an Individual or Community health worker workflow before assessment; this labels the report without changing analysis. Voice and speech features, resting motion, gait cadence, reaction time, clock drawing, and an optional five-word timed Memory Assessment with configurable 5–20 second presentation and an optional countdown. It records recalled words, counts, recall percentage, and completion timing without diagnostic scoring.
* **Audio analysis workflow:** WAV, MP3, M4A, OGG, and FLAC inputs are validated (maximum 25 MB and 120 seconds), converted to mono 16 kHz WAV, and shown with a waveform and relative-dB spectrogram. Descriptive voice measurements are separate from the optional sustained-vowel Parkinson's research-model pathway; no individual disease probability is generated.
* **Brain health timeline:** Review dated session cards, completed activities, audio and memory summaries, report downloads, and session details. Available voice pitch variation, gait cadence, reaction time, clock drawing, and Memory Check measurements can be charted and compared across saved sessions. Missing values are not estimated; differences are descriptive only and do not indicate clinical significance or disease progression. Demo-account history is explicitly marked synthetic.
* **Brain Wellness Companion:** Choose personal wellness goals, receive general sleep, movement, social, cognitive, and routine suggestions, and log daily activities with a seven-day progress view. Plans and check-ins are saved to the signed-in account only after an explicit save; Demo Mode remains session-only. This educational feature does not diagnose, prescribe, prevent or treat disease, or use screening results to determine whether someone is healthy.
* **Professional screening reports:** Export completed sessions as print-ready PDFs or structured JSON. Reports include session metadata, audio processing details, recorded features, available experimental voice-model output, memory results when available, saved history charts, optional saved wellness guidance, limitations, privacy details, and clinician-review guidance. Participant identity is omitted unless explicitly permitted; missing data is not estimated, and reports are not diagnoses.
* **Privacy & Responsible AI:** Assessment controls require an explicit consent checkbox and a separate Continue action. A dedicated page explains collection, processing, storage, limitations, and Demo Mode. Users can clear uploaded inputs and retained audio without discarding completed derived results. PostgreSQL connections require SSL; local SQLite is not encrypted at rest, and production security depends on hosting configuration.
* **Multi-Role Views & Storage:** Includes **Individual**, **Caregiver**, and **Admin Triage** views. Account and finalized-report storage uses PostgreSQL when configured or local SQLite otherwise; unfinished derived-result drafts use a separate local SQLite database. These prototype role checks are not production access controls.

## Hackathon Demo Mode
Select **Enter Demo Mode** on the sign-in screen to explore the application without creating an account. Demo Mode uses a synthetic participant and clearly labelled sample history, keeps new reports in the current Streamlit session only, and does not initialize or access persistent account storage. The generated voice sample is processed by the existing descriptive audio pipeline; it is not a fabricated model result. Recording, arbitrary audio uploads, and participant-entered speech are disabled in this isolated flow. Use **Reset demo data** on the dashboard to restore the initial synthetic history and clear demo activity.

---

## Technical Validation & Research Models
NeuroNexus incorporates a calibrated **Research Model Registry** validated with strict participant-grouped cross-validation:

| Research Benchmark | Dataset | Validation Method | Key Metrics |
| :--- | :--- | :--- | :--- |
| **Alzheimer’s Kinematic Classifier** | UCI DARWIN (174 participants) | 5-Fold `StratifiedGroupKFold` | **0.921 ROC-AUC**, **84.5% Accuracy**, 82.0% Sensitivity, 87.1% Specificity |
| **Parkinson’s Acoustic Classifier** | UCI Oxford Parkinson's (195 recordings) | Participant-Grouped Stratified CV | **0.617 Balanced Accuracy** |

*Note: These classifiers are maintained strictly as cohort-level research benchmarks due to sensor and schema variations in consumer hardware.*

All measurements are exploratory and descriptive. The app does not diagnose, estimate disease risk, or determine treatment. Seek immediate medical care for sudden new neurological symptoms.

---

## Project Structure & Local Setup
The codebase is modularly organized across `app.py`, training pipelines in the root directory, core logic inside `modules/`, models in `models/`, and automated tests in `tests/`.

### Run Locally
```bash
git clone https://github.com
cd NeuroNexus
python -m venv .venv
# Activate venv (.venv\Scripts\Activate.ps1 on Windows or source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Run Automated Tests
```bash
python -m pytest -v
```

---

## Scientific Limitations & Author
* **Limitations:** Consumer microphone variance, ambient noise, and browser loops impact measurements. Prospective multi-site clinical validation is required prior to formal deployment.
* **Author:** **Rimla Shehad** — Lead AI & Full-Stack Software Engineer (Solo Builder, *Future Health Digital Innovation Challenge 2026*).
