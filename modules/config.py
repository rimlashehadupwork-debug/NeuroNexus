"""Central metadata and research-safety language."""
APP_NAME = "NeuroNexus"
APP_VERSION = "1.0.0"
APP_SUBTITLE = "Multimodal digital-biomarker research prototype"
SUPPORTED_MODES = ("Individual", "Community health worker")
DIAGNOSTIC_DISCLAIMER = (
    "Research prototype only — not a diagnostic device. NeuroNexus cannot diagnose, "
    "rule out, or estimate the probability of Alzheimer's disease or Parkinson's disease."
)
CONSENT_TEXT = (
    "I have read the information above and voluntarily consent to analyzing the signal "
    "I provide in this session. I understand this is experimental research software, "
    "not a diagnosis, and I can stop or clear the session at any time."
)
PRIVACY_NOTICE = """
**What this prototype does**
- Analyzes provided audio or motion data in the current app session.
- Does not intentionally save raw uploads to project files or a database.
- Offers a reset control that clears app-held result references.

**Important limits**
- This is not a guarantee of zero retention. Runtime memory, browser state, hosting,
  reverse proxies, logs, monitoring and backups are separate systems and need review.
- Do not upload names, patient identifiers, or data you are not authorized to use.
- Consent in the interface does not replace research ethics review, legal compliance,
  a data-governance plan, or informed participant procedures.
"""
