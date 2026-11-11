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
PRIVACY_NOTICE = """
**What this prototype does**
- Requires a consent checkbox before the assessment controls are shown. This is a
  UI gate, not a validated electronic-consent workflow or a stored consent receipt.
- Processes standard audio and sensor inputs in the active app session; normalized
  voice audio may remain in Streamlit session memory until reset or sign-out.
- When optional matching speech audio is analyzed, a temporary WAV is written to
  the operating system's temporary directory and removed when processing exits.
- Stores account details and saved derived session summaries in local SQLite.
  Raw voice recordings and raw sensor arrays are not intentionally saved there.
- Optional age/sex/device context and caregiver observation checkboxes remain in
  the active session unless the user explicitly saves a report while local session
  history is enabled. The report marks caregiver observations as user-reported.
- The clear-session and sign-out actions clear app-held assessment state. The
  profile's Purge My Session Data action also deletes that account's saved session
  summaries from local SQLite. Neither action can erase files already downloaded.
- Does not automatically strip names, addresses, or other identifying details.
  Do not speak or type personal identifiers into recordings or transcripts.

**Important limits**
- This is not a guarantee of zero retention. Runtime memory, browser state, hosting,
  reverse proxies, logs, monitoring and backups are separate systems and need review.
- Consent in the interface does not replace research ethics review, legal compliance,
  a data-governance plan, or informed participant procedures.
- The local SQLite database is not encrypted at rest. Passwords are salted and
  hashed, but authentication and role checks are demonstration-only, not production
  security.
"""
