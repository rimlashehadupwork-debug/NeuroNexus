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
    "I have read the information above and voluntarily consent to analyzing the signal "
    "I provide in this session. I may also choose to include optional profile context "
    "or caregiver observations in the session report. I understand this is experimental "
    "research software, not a diagnosis, and I can stop or clear the session at any time."
)
STORAGE_ARCHITECTURE_NOTE = (
    "With DATABASE_URL configured in Streamlit secrets or the environment, account "
    "profiles, role and consent flags, and finalized reports use Supabase PostgreSQL. "
    "Set it to a PostgreSQL connection string, not a Supabase publishable API key. "
    "Without it, those records use local SQLite. Unfinished assessment progress is "
    "kept only in a local SQLite draft and is removed when the session is reset, "
    "signed out, or finalized and saved. Optional NEURONEXUS_ADMIN_EMAIL and "
    "NEURONEXUS_ADMIN_PASSWORD secrets provision the first cloud admin. Drafts "
    "contain derived module results, not raw audio or sensor uploads."
)
PRIVACY_NOTICE = """
**What this prototype does**
- Requires a consent checkbox before the assessment controls are shown. This is a
  UI gate; the registration acknowledgment and session consent flags are stored,
  but this is not a validated electronic-consent workflow or receipt.
- Processes standard audio and sensor inputs in the active app session; normalized
  voice audio may remain in Streamlit session memory until reset or sign-out.
- When optional matching speech audio is analyzed, a temporary WAV is written to
  the operating system's temporary directory and removed when processing exits.
- Stores account details, roles, consent flags, and explicitly saved derived
  reports in Supabase PostgreSQL when DATABASE_URL is configured, otherwise in
  local SQLite. Incomplete derived module results are saved only in a local SQLite
  draft to support resuming; raw audio and sensor uploads are not saved in drafts.
- Optional age/sex/device context and caregiver observation checkboxes remain in
  the active session unless the user explicitly saves a report while saved session
  history is enabled. The report marks caregiver observations as user-reported.
- The clear-session, sign-out, and report-save actions remove the local draft.
  Purge My Session Data also deletes that account's finalized reports from the
  configured persistent store. Neither action can erase files already downloaded.
- Does not automatically strip names, addresses, or other identifying details.
  Do not speak or type personal identifiers into recordings or transcripts.

**Important limits**
- This is not a guarantee of zero retention. Runtime memory, browser state, hosting,
  reverse proxies, logs, monitoring and backups are separate systems and need review.
- Consent in the interface does not replace research ethics review, legal compliance,
  a data-governance plan, or informed participant procedures.
- Local SQLite drafts are not encrypted at rest. Passwords are salted and hashed,
  but authentication and role checks are demonstration-only, not production
  security. Review cloud database access policies, backups, logs, and retention
  separately before handling real participant data.
"""
