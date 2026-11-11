"""Descriptive speech features and a separately registered DARWIN research model."""
from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DARWIN_MODEL_PATH = MODELS_DIR / "alzheimers_darwin_research.joblib"
DARWIN_METADATA_PATH = MODELS_DIR / "alzheimers_darwin_research_metadata.json"
DEFAULT_DARWIN_DATASET = PROJECT_ROOT / "data" / "darwin_data.csv"
DARWIN_CITATION = (
    "Fontanella, F. (2022). DARWIN [Dataset]. UCI Machine Learning Repository."
)
DARWIN_DOI = "10.24432/C55D0K"
DARWIN_URL = "https://archive.ics.uci.edu/dataset/732/darwin"
DARWIN_FEATURE_COUNT = 450
DARWIN_TOP_K = 20
DARWIN_LABELS = {"H": 0, "P": 1}


def get_alzheimers_status() -> dict:
    """Return registry status for the DARWIN kinematic research artifact."""
    if not (DARWIN_MODEL_PATH.is_file() and DARWIN_METADATA_PATH.is_file()):
        return {
            "available": False,
            "dataset_status": "darwin_model_not_trained",
            "message": (
                "The DARWIN kinematic research model is not installed. "
                "No individual risk result is calculated."
            ),
            "validation_stage": "none",
        }
    meta = json.loads(DARWIN_METADATA_PATH.read_text(encoding="utf-8"))
    metrics = meta.get("metrics") or {}
    return {
        "available": True,
        "model_name": meta.get("model_name", "Alzheimer's DARWIN kinematic research model"),
        "version": meta.get("version"),
        "dataset_source": (meta.get("dataset") or {}).get("source", "uci_darwin"),
        "participants": (meta.get("dataset") or {}).get("participant_count"),
        "recording_level": None,
        "subject_level": metrics.get("participant_level") or {},
        "subject_level_95ci": metrics.get("participant_balanced_accuracy_95ci"),
        "selected_model": meta.get("selected_model"),
        "trained_on": meta.get("trained_on"),
        "validation_stage": meta.get("validation_stage", 2),
        "limitations": meta.get("limitations", []),
        "message": (
            "DARWIN kinematic research artifact installed; internal grouped CV only. "
            "It is not used for live canvas scoring or individual risk prediction."
        ),
        "metadata": meta,
    }


def get_alzheimers_speech_status() -> dict:
    """The transcript pathway remains descriptive and has no trained speech model."""
    return {
        "available": False,
        "dataset_status": "no_compatible_speech_model",
        "message": (
            "No Alzheimer's speech model is installed; the DARWIN artifact is kinematic only. "
            "Transcript measurements are descriptive, with no individual risk result."
        ),
        "validation_stage": "none",
    }


def extract_speech_features(transcript: str, audio_path: str | None = None) -> dict:
    """Extract the planned Alzheimer's speech feature set.

    Lexical features come from the transcript alone (deterministic, no model,
    no external NLP dependency). Prosodic features come from the audio file
    when provided (16 kHz mono WAV or any format `to_standard_wav` decodes).
    Every feature is documented and reproducible. These are research features,
    not diagnostic outputs.
    """
    if not transcript or not transcript.strip():
        raise ValueError("A transcript (even a rough manual one) is required for the lexical features.")
    tokens = re.findall(r"[a-z'\-]+", transcript.lower())
    n_words = len(tokens)
    if n_words < 10:
        raise ValueError("Transcript is too short (<10 tokens) for stable lexical features.")
    types = set(tokens)
    n_pron = sum(t in _PRONOUNS for t in tokens)
    n_content = sum(t in _CONTENT_TAGS or _is_content_word(t) for t in tokens)

    features: dict = {
        "n_words": n_words,
        "n_unique_words": len(types),
        "type_token_ratio": round(len(types) / n_words, 5),
        "pronoun_ratio": round(n_pron / n_words, 5),
        "idea_density": round(n_content / n_words, 5),  # content words / total words
        "mean_word_length": round(sum(len(t) for t in tokens) / n_words, 5),
        "n_repeated_bigrams": sum(1 for a, b in zip(tokens, tokens[1:]) if a == b),
    }

    if audio_path is not None:
        path = Path(audio_path)
        raw = path.read_bytes()
        from .audio_analysis import to_standard_wav
        wav_bytes, _ = to_standard_wav(raw)
        import io
        import numpy as np, librosa, soundfile as sf
        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        duration = float(len(audio) / sr)
        features["duration_seconds"] = round(duration, 2)
        if duration > 0:
            features["words_per_minute"] = round(n_words / (duration / 60.0), 2)
        # Silence-based pause ratio: fraction of 25 ms frames below an RMS threshold.
        frame_len = max(1, int(sr * 0.025))
        frames = [audio[i:i+frame_len] for i in range(0, len(audio), frame_len) if len(audio[i:i+frame_len])]
        frame_rms = np.array([np.sqrt(np.mean(f**2)) for f in frames])
        floor = max(0.001, float(np.percentile(frame_rms, 20)))
        features["pause_ratio"] = round(float(np.mean(frame_rms < floor)), 5)
        features["speech_rate_variability"] = round(float(np.std(frame_rms)) , 6)
        mfcc = librosa.feature.mfcc(y=audio.astype(np.float32), sr=int(sr), n_mfcc=13)
        features["mfcc_means"] = [round(float(v), 5) for v in mfcc.mean(axis=1)]
        features["mfcc_stds"] = [round(float(v), 5) for v in mfcc.std(axis=1)]
    return features


_PRONOUNS = {
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves",
}
_CONTENT_TAGS = {"n/a"}  # reserved; heuristic below is the working method
_FUNC_PREFIXES = ("un", "re", "dis", "pre", "over", "under")  # weak derivational signal


def _is_content_word(tok: str) -> bool:
    """Weak, transparent heuristic: content words are open-class (nouns, verbs,
    adjectives, adverbs). We approximate with length + suffix rules so the
    method stays reproducible without an NLP library."""
    if len(tok) <= 3:
        return False
    if tok.endswith(("ly", "tion", "ness", "ment", "ity", "ing", "ed", "er", "est", "ful", "ous", "ive", "able", "ible", "al", "ish")):
        return True
    return len(tok) >= 6  # longer closed-class words are rare; keep transparent


def run_alzheimers_feature_report(transcript: str, audio_path: str | None = None) -> dict:
    """UI-facing speech entry point; DARWIN kinematics are not a speech model."""
    features = extract_speech_features(transcript, audio_path)
    status = get_alzheimers_speech_status()
    return {
        "features": features,
        "audio_used": audio_path is not None,
        "model_available": status["available"],
        "message": status.get("message", ""),
        "note": ("Feature extraction is descriptive only. No compatible, independently validated "
                 "speech model is installed, so no individual risk score is calculated."),
    }


def load_darwin_dataset(path=DEFAULT_DARWIN_DATASET):
    """Load the labeled DARWIN table with participant IDs kept separate from features."""
    df = pd.read_csv(path)
    normalized = {str(column).strip().lower(): column for column in df.columns}
    if "id" not in normalized or "class" not in normalized:
        raise ValueError("DARWIN CSV must contain ID and class columns.")
    id_column, target_column = normalized["id"], normalized["class"]
    feature_columns = [
        column for column in df.columns
        if column not in (id_column, target_column)
    ]
    if len(feature_columns) != DARWIN_FEATURE_COUNT:
        raise ValueError(
            f"Expected {DARWIN_FEATURE_COUNT} DARWIN feature columns, found {len(feature_columns)}."
        )
    if df[id_column].isna().any():
        raise ValueError("DARWIN ID values must be present for participant-grouped validation.")
    groups = df[id_column].astype(str).str.strip().to_numpy()
    if any(not group for group in groups):
        raise ValueError("DARWIN ID values must not be empty.")
    labels = df[target_column].astype(str).str.strip().str.upper()
    unknown_labels = sorted(set(labels) - set(DARWIN_LABELS))
    if unknown_labels:
        raise ValueError(f"Unexpected DARWIN class labels: {unknown_labels}. Expected H and P.")
    y = labels.map(DARWIN_LABELS).to_numpy(dtype=int)
    X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    values = X.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("DARWIN feature columns must contain only finite numeric values.")
    group_labels = pd.DataFrame({"group": groups, "target": y}).groupby("group")["target"].nunique()
    if (group_labels > 1).any():
        raise ValueError("Each DARWIN participant ID must map to exactly one class label.")
    return X, y, groups, feature_columns


def _darwin_estimator(top_k=DARWIN_TOP_K, calibration_cv=3, random_state=42):
    base = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("selector", SelectKBest(score_func=f_classif, k=top_k)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            random_state=random_state,
        )),
    ])
    return CalibratedClassifierCV(
        estimator=base,
        method="sigmoid",
        cv=calibration_cv,
        ensemble=True,
    )


def _darwin_metrics(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "sensitivity": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, predictions, pos_label=0, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": {
            "true_healthy": int(tn),
            "false_patient": int(fp),
            "false_healthy": int(fn),
            "true_patient": int(tp),
        },
    }


def _confidence_interval(values):
    if len(values) < 2:
        return None
    from scipy.stats import sem, t

    low, high = t.interval(
        0.95,
        len(values) - 1,
        loc=float(np.mean(values)),
        scale=float(sem(values)),
    )
    return [float(low), float(high)]


def train_darwin_model(
    dataset_path=DEFAULT_DARWIN_DATASET,
    output_dir=MODELS_DIR,
    n_splits=5,
    top_k=DARWIN_TOP_K,
    calibration_cv=3,
    random_state=42,
):
    """Train a calibrated research model with participant-grouped outer CV."""
    X, y, groups, feature_columns = load_darwin_dataset(dataset_path)
    if not 1 <= top_k <= X.shape[1]:
        raise ValueError(f"top_k must be between 1 and {X.shape[1]}.")
    class_group_counts = {
        label: len(np.unique(groups[y == target]))
        for label, target in (("H", 0), ("P", 1))
    }
    n_splits = min(n_splits, min(class_group_counts.values()))
    if n_splits < 3:
        raise ValueError("At least three participant IDs per class are required for grouped CV.")
    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    oof_probabilities = np.full(len(y), np.nan, dtype=float)
    fold_metrics = []
    for train_indices, test_indices in cv.split(X, y, groups):
        estimator = _darwin_estimator(top_k, calibration_cv, random_state)
        estimator.fit(X.iloc[train_indices], y[train_indices])
        probabilities = estimator.predict_proba(X.iloc[test_indices])[:, 1]
        oof_probabilities[test_indices] = probabilities
        fold_metrics.append(_darwin_metrics(y[test_indices], probabilities))
    if not np.isfinite(oof_probabilities).all():
        raise RuntimeError("Cross-validation did not produce an out-of-fold probability for every row.")

    participant_metrics = _darwin_metrics(y, oof_probabilities)
    mean_metrics = {
        key: float(np.mean([fold[key] for fold in fold_metrics]))
        for key in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "roc_auc", "brier_score")
    }
    std_metrics = {
        key: float(np.std([fold[key] for fold in fold_metrics], ddof=1))
        for key in mean_metrics
    }
    f_scores, p_values = f_classif(X, y)
    ranked_indices = np.argsort(np.nan_to_num(f_scores, nan=-np.inf))[::-1][:top_k]
    selected_features = [feature_columns[index] for index in ranked_indices]
    feature_ranking = [
        {
            "name": feature_columns[index],
            "f_score": float(f_scores[index]) if np.isfinite(f_scores[index]) else None,
            "p_value": float(p_values[index]) if np.isfinite(p_values[index]) else None,
        }
        for index in ranked_indices
    ]
    cohort_median_profile = {
        feature: float(X[feature].median())
        for feature in selected_features
    }

    final_estimator = _darwin_estimator(top_k, calibration_cv, random_state)
    final_estimator.fit(X, y)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / DARWIN_MODEL_PATH.name
    metadata_path = out_dir / DARWIN_METADATA_PATH.name
    joblib.dump({
        "pipeline": final_estimator,
        "feature_columns": list(feature_columns),
        "group_column": "ID",
        "target_column": "class",
        "target_mapping": dict(DARWIN_LABELS),
        "selected_feature_names_full_fit": selected_features,
        "model_type": "Calibrated Logistic Regression",
    }, model_path)

    participant_counts = {
        label: int(len(np.unique(groups[y == target])))
        for label, target in (("H", 0), ("P", 1))
    }
    metadata = {
        "model_name": "Alzheimer's DARWIN kinematic research model",
        "version": "1.0.0",
        "status": "experimental_research_only_no_patient_predictions",
        "dataset_rows": int(len(X)),
        "participant_count": int(len(np.unique(groups))),
        "feature_count": int(X.shape[1]),
        "dataset": {
            "source": "uci_darwin",
            "name": "DARWIN Handwriting/Drawing Dataset",
            "citation": DARWIN_CITATION,
            "doi": DARWIN_DOI,
            "url": DARWIN_URL,
            "license": "CC BY 4.0",
            "rows": int(len(X)),
            "participant_count": int(len(np.unique(groups))),
            "class_participant_counts": {
                "healthy_H": participant_counts["H"],
                "patient_P": participant_counts["P"],
            },
            "feature_count": int(X.shape[1]),
            "features": list(feature_columns),
            "cohort_median_profile": cohort_median_profile,
        },
        "group_column": "ID",
        "target_column": "class",
        "target_mapping": dict(DARWIN_LABELS),
        "selected_model": "Calibrated Logistic Regression",
        "feature_selection": {
            "method": "SelectKBest(f_classif)",
            "k": int(top_k),
            "selected_features": selected_features,
            "ranking": feature_ranking,
        },
        "preprocessing": {
            "imputation": "SimpleImputer(strategy='median')",
            "scaling": "StandardScaler",
            "calibration": {
                "method": "sigmoid",
                "folds": int(calibration_cv),
            },
        },
        "metrics": {
            "evaluation": (
                f"{n_splits}-fold StratifiedGroupKFold; ID kept as the group; "
                "each row represents one participant in this dataset."
            ),
            "participant_level": participant_metrics,
            "accuracy": participant_metrics["accuracy"],
            "balanced_accuracy": participant_metrics["balanced_accuracy"],
            "sensitivity": participant_metrics["sensitivity"],
            "specificity": participant_metrics["specificity"],
            "roc_auc": participant_metrics["roc_auc"],
            "brier_score": participant_metrics["brier_score"],
            "precision": participant_metrics["precision"],
            "f1": float(
                2 * participant_metrics["precision"] * participant_metrics["sensitivity"]
                / max(
                    participant_metrics["precision"] + participant_metrics["sensitivity"],
                    1e-12,
                )
            ),
            "confusion_matrix": participant_metrics["confusion_matrix"],
            "fold_mean": mean_metrics,
            "fold_standard_deviation": std_metrics,
            "participant_balanced_accuracy_95ci": _confidence_interval(
                [fold["balanced_accuracy"] for fold in fold_metrics]
            ),
        },
        "trained_on": datetime.now(timezone.utc).isoformat(),
        "validation_stage": 2,
        "limitations": [
            "Small retrospective dataset (174 participants); cross-validation is internal and not external validation.",
            "Class labels are dataset labels, not clinical diagnoses or an estimate of disease probability.",
            "The 450-feature vector aggregates 25 instrumented tasks and is not equivalent to a single browser clock drawing.",
            "The Streamlit canvas has no calibrated pressure or reliable pointer timestamps; air time, drawing duration, stroke speed, and jerk cannot be recovered.",
            "The research artifact is not used for live canvas scoring or individual risk prediction.",
            "No prospective, demographic fairness, clinical utility, or regulatory validation has been performed.",
        ],
        "model_file": str(model_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    return metadata


def train_alzheimers_model(dataset_path=DEFAULT_DARWIN_DATASET, output_dir=MODELS_DIR, **kwargs):
    """Backward-compatible entry point for the DARWIN kinematic research trainer."""
    source = Path(dataset_path)
    if source.is_dir():
        source = source / "darwin_data.csv"
    return train_darwin_model(source, output_dir=output_dir, **kwargs)


def load_darwin_sample_profile() -> dict:
    """Return a cohort-median feature example, never an individual or prediction."""
    status = get_alzheimers_status()
    if not status.get("available"):
        raise RuntimeError(status["message"])
    metadata = status["metadata"]
    return {
        "profile_name": "DARWIN cohort median feature profile",
        "features": metadata["dataset"]["cohort_median_profile"],
        "model_version": metadata.get("version"),
        "notice": (
            "This is a dataset-wide median of 174 participants, not an individual "
            "or held-out test case. No model inference was performed."
        ),
    }
