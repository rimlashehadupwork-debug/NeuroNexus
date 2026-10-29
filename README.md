# NeuroNexus

**Version:** 0.1.0  
A privacy-conscious multimodal digital-biomarker **research prototype** for exploring voice and smartphone motion signals.

## Day 1–3 foundation

- Centralized app metadata and research-safety disclaimer (`modules/config.py`)
- Session initialization that does not overwrite an active session (`modules/session.py`)
- Explicit consent gate before sample analysis
- Individual and Community Health Worker workflows
- Data-handling notice visible in the sidebar
- Session reset clears app-held analysis results and rotates upload/consent widget keys
- Unit tests for audio, motion, and session lifecycle

## Current features

- Streamlit dashboard with individual and community health worker modes
- WAV upload and in-memory acoustic feature extraction
- CSV-based acceleration analysis with FFT visualization
- Basic signal-quality flags
- Session result clearing

## Not a medical diagnostic tool

This project does not diagnose, rule out, or calculate the probability of Alzheimer's disease or Parkinson's disease. Frequency peaks and voice features are nonspecific. Clinical use would require appropriately designed studies, representative datasets, external validation, and relevant regulatory review.

## Setup

Python 3.10–3.12 is recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks environment activation, run `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\python.exe -m streamlit run app.py`.

## Run tests

From the `NeuroNexus` project folder:

```powershell
python -m pytest -q
```

## Motion CSV format

Use numeric columns `timestamp,ax,ay,az`, where timestamps are seconds and strictly increasing. If timestamps are omitted, the demo assumes 50 Hz; this assumption must be visible in any report.

## Privacy notes

The app does not intentionally write raw uploads to project files. The reset control clears app-held analysis-result references and rotates the upload/consent widget keys so the UI starts a new app-level session. This does **not** guarantee immediate physical memory erasure or clear browser, hosting, proxy, telemetry, error-reporting, or infrastructure logs. Review these systems before deployment. Do not upload identifiable or clinically sensitive data.

Consent is a user-interface safeguard, not a substitute for formal research ethics review, participant information, data governance, or applicable legal requirements.

## Next milestones

1. Refine visual design and mode-specific workflow
2. Browser microphone capture
3. Mobile sensor capture via JavaScript and HTTPS
4. Gait event detection and confidence checks
5. Research validation plan
