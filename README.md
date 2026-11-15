# NeuroNexus — To Sense is to Predict
**A 3-Minute Multimodal Early Neuro-Screening & Primary Care Triage Prototype**

🚀 **Live Application:** [neuronexus18.streamlit.app](https://streamlit.app)  
🏆 **Built for:** Future Health Digital Innovation Challenge (Co-created by Microsoft, Inception, and MBZUAI)

---

## Overview
**NeuroNexus** is a browser-based multimodal neuro-screening and primary care triage platform requiring zero specialized hardware. It captures six digital biomarkers in under three minutes, translates findings into plain English, and generates a downloadable **Clinician Summary PDF** to support faster primary care referrals for neurodegenerative conditions like Parkinson’s and Alzheimer’s.

> **Medical & Research Disclaimer:** NeuroNexus is an exploratory research and triage prototype. It does **not** diagnose, rule out, or estimate the clinical probability of Alzheimer’s disease, Parkinson’s disease, or any other neurological condition.

---

## Key Capabilities
* **Six Guided Digital Biomarker Modules (< 3 Minutes):** Includes Voice Acoustic Stability, Speech Fluency, Resting Motion & Tremor (FFT spectral peaks), Gait Cadence & Variability, Interactive Clock Drawing (stroke kinematics), and Reaction-Time Variability.
* **Plain-English & Clinician-Ready Reporting:** Translates metrics into accessible summaries and generates automated **Clinician Summary PDFs** and structured JSON exports.
* **Multi-Role Governance & Hybrid Privacy Storage:** Features **Individual**, **Caregiver**, and **Admin Triage** views backed by a hybrid cloud PostgreSQL (Supabase) and ephemeral SQLite architecture with consent enforcement and one-click session purging.

---

## Technical Validation & Research Models
NeuroNexus incorporates a calibrated **Research Model Registry** validated with strict participant-grouped cross-validation:

| Research Benchmark | Dataset | Validation Method | Key Metrics |
| :--- | :--- | :--- | :--- |
| **Alzheimer’s Kinematic Classifier** | UCI DARWIN (174 participants) | 5-Fold `StratifiedGroupKFold` | **0.921 ROC-AUC**, **84.5% Accuracy**, 82.0% Sensitivity, 87.1% Specificity |
| **Parkinson’s Acoustic Classifier** | UCI Oxford Parkinson's (195 recordings) | Participant-Grouped Stratified CV | **0.617 Balanced Accuracy** |

*Note: These classifiers are maintained strictly as cohort-level research benchmarks due to sensor and schema variations in consumer hardware.*

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
