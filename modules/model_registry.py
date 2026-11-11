"""Model registry: reads metadata JSONs for the admin dashboard.

Every disease pathway reports: model availability, dataset source, participant
counts, grouped-CV metrics with CIs, training date, and validation stage
(1 trained / 2 held-out participants / 3 external / 4 clinical).
"""
from __future__ import annotations
import json
from pathlib import Path
import joblib

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _load_meta(name):
    p = MODELS_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def get_parkinson_status() -> dict:
    meta = _load_meta("parkinson_voice_research_metadata.json")
    model_exists = (MODELS_DIR / "parkinson_voice_research.joblib").exists()
    if not (meta and model_exists):
        return {"available": False, "message": "Parkinson's research model artifact not found."}
    m = meta["metrics"]
    return {
        "available": True,
        "model_name": meta["model_name"],
        "version": meta.get("version"),
        "dataset_source": meta["dataset"]["source"],
        "participants": meta["dataset"]["participant_count"],
        "recording_level": m["recording_level"],
        "subject_level": m["subject_level_mean_vote"],
        "subject_level_95ci": m.get("subject_level_balanced_acc_95ci"),
        "selected_model": m["selected_model"],
        "models_compared": m["models_compared"],
        "trained_on": meta.get("trained_on"),
        "validation_stage": 2,  # evaluated on held-out participants (grouped CV); not externally/clinically validated
        "limitations": meta.get("limitations", []),
    }


def get_registry() -> dict:
    """Return research artifact status; Alzheimer's refers to DARWIN kinematics.

    Descriptive Alzheimer’s transcript features are not a trained model.
    """
    from .alzheimers_model import get_alzheimers_status
    return {"parkinsons": get_parkinson_status(), "alzheimers": get_alzheimers_status()}


def load_parkinson_model():
    """Load artifact for inference on compatible 26-feature vectors only."""
    bundle = joblib.load(MODELS_DIR / "parkinson_voice_research.joblib")
    return bundle["pipeline"], bundle["feature_columns"], bundle.get("threshold", 0.5)
