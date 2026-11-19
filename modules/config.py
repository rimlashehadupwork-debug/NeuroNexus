"""Central metadata and research-safety language."""
APP_NAME = "NeuroNexus"
APP_VERSION = "2.0.0"
APP_SUBTITLE = "Multimodal digital-biomarker research prototype"
SUPPORTED_MODES = ("Individual", "Community health worker")
DIAGNOSTIC_DISCLAIMER = (
    "Research prototype only — not a diagnostic device. NeuroNexus cannot diagnose, "
    "rule out, or estimate the probability of Alzheimer's disease or Parkinson's disease."
)
CONSENT_TEXT = (
    "I have read and understood the information above. I voluntarily agree to process "
    "the assessment information I choose to provide for this session, and understand "
    "that I may stop at any time. This prototype does not provide a diagnosis."
)
CONSENT_INFORMATION = (
    (
        "Information collected",
        "Depending on the activities you choose, this may include voice recordings, "
        "optional matching speech audio and transcript text, motion or gait CSV files, "
        "clock-drawing strokes, reaction-time taps, and words you recall in the memory "
        "task. Optional age range, device/hand preference, and participant-approved "
        "caregiver notes may also be used. Do not include names or other identifiers "
        "in recordings, transcripts, or uploads.",
    ),
    (
        "Purpose and processing",
        "Inputs are processed by this app to produce descriptive signal measurements "
        "and task summaries. Audio is decoded and may be converted to mono 16 kHz WAV. "
        "Optional speech-audio processing uses a temporary file that is removed when "
        "processing exits. No raw audio or sensor upload is included in saved reports "
        "or assessment drafts.",
    ),
    (
        "Results and limitations",
        "You may see descriptive measurements, technical recording-quality checks, "
        "and, when enabled, an experimental research-model output. These do not "
        "diagnose, rule out, or estimate the likelihood of disease. Results can vary "
        "with recording conditions, device, language, accent, task, fatigue, and other "
        "factors. Discuss health concerns with a qualified healthcare professional.",
    ),
    (
        "Storage and retention",
        "For signed-in accounts, unfinished derived results may be saved to a local "
        "SQLite draft after consent. A finalized derived report is stored only when "
        "you explicitly save it. Account/report storage uses PostgreSQL when configured "
        "or local SQLite otherwise; local SQLite is not encrypted at rest. Audio and "
        "other uploaded files can remain in the active app session until cleared, "
        "signed out, or the runtime session ends. Hosting logs, backups, and platform "
        "telemetry are outside this app's retention controls.",
    ),
    (
        "Your choice and how to exit",
        "Participation is optional. Choose “Not now” to return to the dashboard. You "
        "can stop any activity, clear the current screening session, or sign out. "
        "Clearing or signing out removes the active draft but does not delete reports "
        "you previously saved; those can be removed from My Profile.",
    ),
    (
        "Consent and Demo Mode",
        "The assessment page requires a separate Continue action after the consent "
        "checkbox. Only a consent flag is stored; no signed consent receipt or "
        "acceptance timestamp is created. Demo Mode is explicitly labeled, uses "
        "synthetic history and generated sample audio, and keeps activity in the "
        "current Streamlit session instead of account storage.",
    ),
)
RESPONSIBLE_AI_NOTICE = (
    "NeuroNexus is a research-support prototype, not a diagnostic system. Its models "
    "and measurements have important technical and validation limitations; outputs "
    "are not clinical conclusions. Recording conditions, speech characteristics, "
    "language, devices, and other factors can influence results. Seek professional "
    "clinical evaluation for health concerns, especially if symptoms persist or "
    "worsen."
)
STORAGE_ARCHITECTURE_NOTE = (
    "With DATABASE_URL configured in Streamlit secrets or the environment, account "
    "profiles, role and consent flags, and finalized reports use Supabase PostgreSQL. "
    "PostgreSQL connections require SSL transport (sslmode=require or stronger). "
    "Set it to a PostgreSQL connection string, not a Supabase publishable API key. "
    "Without it, those records use local SQLite. Unfinished assessment progress is "
    "kept only in a local SQLite draft after consent and is removed when the session is reset, "
    "signed out, or finalized and saved. Optional NEURONEXUS_ADMIN_EMAIL and "
    "NEURONEXUS_ADMIN_PASSWORD secrets provision the first cloud admin. Drafts "
    "contain derived module results, not raw audio or sensor uploads."
)
PRIVACY_NOTICE = """
**What this prototype does**
- Assessment controls require an explicit consent checkbox and a separate Continue
  action. This is a user-interface gate, not a validated electronic-consent workflow
  or a consent receipt; the account stores only a consent flag.
- The app does not load an analytics tag or externally hosted fonts. Hosting-provider
  access logs, backups, and platform telemetry are outside the app's control.
- Audio and sensor files are processed in the active app session. Audio can remain
  available in session memory until cleared or the session ends. Optional matching
  speech audio is written to a temporary directory that is removed when processing
  exits. The assessment page provides a control to clear uploaded inputs and retained
  audio while keeping completed derived measurements.
- Signed-in unfinished derived results are saved to a local SQLite draft after
  consent; raw uploads are not included. Finalized derived reports and account data
  use PostgreSQL when DATABASE_URL is configured, otherwise local SQLite. PostgreSQL
  connections require SSL transport. Local SQLite is not encrypted at rest.
- Demo Mode uses synthetic data and keeps activity in the current Streamlit session;
  it does not use account storage.
- Clearing a session or signing out removes the active draft, not previously saved
  reports or files already downloaded. Use “Purge My Session Data” in My Profile to
  delete that account's saved assessment reports.
- The app does not automatically detect or redact names, addresses, or other
  identifying information. Avoid including identifiers in audio, transcripts, or
  uploaded files.

**Important limits**
- HTTPS for the app itself depends on the hosting platform or reverse proxy. PostgreSQL
  SSL transport does not replace certificate verification when configured as `require`.
- Passwords are salted and PBKDF2-hashed, but authentication and role checks are
  demonstration-only, not production security. Caregiver report access is matched
  against the email entered in a participant profile; address ownership is not
  verified. Review server-side authorization, database policies, encryption,
  backups, logs, and retention before handling real participant data.
- Consent does not replace research ethics review, lawful-basis assessment, or
  participant procedures. This is not a guarantee of zero retention.
"""
