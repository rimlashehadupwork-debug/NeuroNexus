> **🚀 Live application:** [Open NeuroNexus 2.0](https://neuronexus18.streamlit.app/)

# NeuroNexus 2.0 — Multimodal Research Prototype

NeuroNexus 2.0 (NeuroNexus18) is an interactive Streamlit web application for exploring multimodal digital biomarkers, including voice acoustic analysis, resting tremor FFT motion analysis, gait cadence tracking, clock-drawing tasks, and reaction-time testing. It is a research prototype for exploring signal measurements, not a medical diagnostic tool.

The interface follows the supplied Stitch NeuroBaseline reference: calm clinical minimalism, slate/navy structure, mineral teal accents, generous spacing and legible typography.

> **Medical safety:** This project does not diagnose, rule out, or estimate the probability of Alzheimer's disease or Parkinson's disease. No clinically validated disease classifier is included. Results are exploratory measurements only. A neurologist or other qualified clinician must assess health concerns.

## Included in this build

- **Branding and UI:** NeuroNexus logo, responsive Streamlit layout, clinical-minimal design tokens inspired by the Stitch reference.
- **Authentication demo:** individual, caregiver and administrator roles; salted PBKDF2 password hashes; Supabase PostgreSQL storage when `DATABASE_URL` is configured, otherwise local SQLite.
- **Individual dashboard:** session counts, recent saved summaries and navigation to assessments.
- **Profile:** name and caregiver email linking.
- **Voice analysis:** preserves the existing WAV recording/upload and acoustic feature extraction logic.
- **Resting motion:** preserves existing CSV parsing, signal-quality checks and FFT analysis.
- **Gait analysis:** preserves existing exploratory acceleration-peak and cadence logic.
- **Clock drawing:** canvas task with a clear instruction; no unvalidated cognitive score is calculated.
- **Reaction-time activity:** short visual response task. Browser/server timing is approximate and not a clinical measure.
- **PDF and JSON reports:** derived features and technical quality only; raw recordings and sensor series are not included.
- **Parkinson’s research model laboratory:** accepts the labelled UCI Oxford voice-feature CSV, runs participant-grouped cross-validation, reports exploratory metrics and saves a research model artifact. It is not connected to live patient predictions.
- **Caregiver view:** linked participants' saved session summaries. It does not send automatic disease or cognitive-decline alerts.
- **Admin overview:** account/session counts, user directory and recent session metadata.
- **Documentation:** project usage and explicit research limitations.

## Run on Windows

Python 3.10–3.12 is recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`.

## Demo administrator account

- Email: `admin@neuronexus.local`
- Password: `ChangeMe123!`

Change or remove this seeded demo account before any shared deployment. This authentication/database setup is a demonstration scaffold, not production-grade security. Do not expose it publicly as-is.

## Accounts and caregiver linking

1. Create an Individual account from the sign-in page.
2. If you want a caregiver to see saved summaries, the individual enters the caregiver's registered email in **My profile**.
3. The caregiver creates a separate account and selects **Caregiver**.
4. The caregiver portal displays saved session summaries linked to that email.

This is a prototype relationship mechanism only. A real deployment needs verified identity, explicit revocable consent, least-privilege authorization, audit logs and secure account recovery.

## Screening workflow

1. Sign in with an individual account.
2. Read the consent/data-handling notice and tick consent.
3. Complete any safe, optional assessment: voice, resting motion, gait, clock drawing or reaction time.
4. Open Report to inspect the measurements and download PDF/JSON.
5. In **My profile**, enable local session history if desired; then choose **Save session to my dashboard** to store the derived summary in the configured persistent database (Supabase PostgreSQL when `DATABASE_URL` is configured, otherwise local SQLite).

### Sensor CSV

Preferred columns: `timestamp,ax,ay,az`, timestamps in seconds and strictly increasing. If timestamps are absent, the existing analysis code assumes 50 Hz. The supplied `assets/sensor_logger.html` records accelerometer values in the browser; HTTPS and sensor permissions may be required. It may include gravity and is not a calibrated clinical sensor.

## Data and privacy notes

- Raw audio and raw sensor arrays are processed in app memory and are not intentionally written to the SQLite database.
- Account name, email, role, caregiver email, password hash/salt and explicitly saved **derived session summaries** use persistent Supabase PostgreSQL storage when `DATABASE_URL` is configured (as in the live deployment), falling back to local SQLite in offline mode. In-progress assessment drafts remain in local SQLite.
- Optional participant context and consented caregiver observations remain in the active app session and are included in a saved summary only by explicit action. **Purge My Session Data** deletes this account's saved summaries and current in-memory assessment state, but cannot erase downloaded files or system logs.
- Streamlit runtime memory, browser state, hosting provider, logs, telemetry, backups and downloads require separate review. This is not a guarantee of zero retention or complete erasure.
- Do not enter real patient identifiers or upload data without authorization. Consent in this prototype does not replace ethics review, informed consent procedures or a data governance plan.

## Parkinson’s research model workflow

The admin **Research model laboratory** accepts the UCI Oxford Parkinson’s voice CSV (`name`, numeric voice feature columns, `status`). The training script extracts participant IDs from names such as `phon_R01_S01_1` and uses stratified grouped cross-validation so repeated recordings from one participant are not split across training and validation folds. It reports balanced accuracy, sensitivity, specificity, precision, F1 and ROC-AUC, then saves a model artifact and metadata under `models/`.

**Critical compatibility limit:** this UCI dataset contains precomputed biomedical voice features (jitter, shimmer, HNR and related measures). The current NeuroNexus audio module calculates MFCC summaries, RMS, pitch and spectral measures. These are not the same feature schema. Therefore the research model is intentionally not wired to the live voice assessment and must not be used to generate a user's disease result. A compatible feature-extraction pipeline and independent external validation are required first. The UCI source describes 195 recordings from 31 people (23 with Parkinson's); repeated recordings make participant-level splitting essential. Source: [UCI Oxford Parkinson’s Disease Detection Dataset](https://archive.ics.uci.edu/dataset/174/parkinsons).

## DARWIN kinematic research workflow

Place the UCI DARWIN dataset at `data/darwin_data.csv` and run:

```powershell
python train_alzheimers_darwin.py
```

The script validates the 174-row, 450-feature `ID`/`class` schema, maps `H`/`P` labels, selects 20 features with `SelectKBest(f_classif)`, and trains a sigmoid-calibrated Logistic Regression pipeline with median imputation and `StandardScaler`. Five-fold `StratifiedGroupKFold` keeps participant IDs grouped. It saves `models/alzheimers_darwin_research.joblib` and the matching metadata JSON with dataset attribution, selected features, and internal validation metrics. Attribution: Fontanella (2022), UCI DARWIN, CC BY 4.0, DOI [10.24432/C55D0K](https://doi.org/10.24432/C55D0K).

**Critical compatibility limit:** DARWIN's 450 features summarize 25 instrumented tasks. The current browser clock canvas has no calibrated pressure or pointer timestamps and records only coarse geometry for one task. The research model is registered for dataset-level evaluation, but it is not run on live drawings and does not produce an individual similarity or disease-risk result. The "Sample DARWIN Test Profile" button displays cohort medians only—not a person or held-out test participant—and does not run inference.

## Scientific limitations and next research steps

- Voice measures vary with language, accent, age, task, microphone, room noise and unrelated health factors.
- Motion frequency peaks are nonspecific; the current motion code is not a Parkinson's detector.
- Gait uses an unvalidated acceleration-peak heuristic and can miss or double-count steps.
- Clock drawing is captured but not scored. Reaction timing is affected by device/browser latency.
- Parkinson’s and DARWIN model metrics are small, retrospective internal cross-validation results, not clinical validation. Neither model is used to generate user-level disease predictions; the app must not display disease-positive/negative decisions or disease probabilities.
- Before any disease-specific screening claim: obtain suitable representative clinically labelled data; define a protocol and intended use; obtain ethics and data approvals; train separate models; pre-specify thresholds; measure sensitivity, specificity and calibration; validate independently across devices/demographics; obtain clinical oversight and regulatory review as applicable.

## Test status

The original analysis tests are included. In this build environment, `python -m pytest -q` passed all 7 tests. Streamlit UI dependencies must be installed locally from `requirements.txt` before launching the interface.

## Frequently Asked Questions (FAQ)

### What is NeuroNexus 2.0?

NeuroNexus 2.0 (NeuroNexus18) is an interactive Streamlit research application for exploring multimodal digital biomarkers using voice, motion, gait, clock-drawing, and reaction-time activities.

### What digital biomarkers does NeuroNexus measure?

It explores voice acoustic features, resting tremor motion frequencies using FFT, gait cadence, clock-drawing task measurements, and reaction time. These measurements are exploratory and are not clinically validated diagnostic results.

### Is NeuroNexus a medical diagnostic tool?

No. NeuroNexus does not diagnose, rule out, or estimate the probability of Alzheimer's disease or Parkinson's disease. Its results are exploratory measurements only; health concerns should be assessed by a qualified clinician.
