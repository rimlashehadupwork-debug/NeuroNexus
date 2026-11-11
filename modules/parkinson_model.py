"""Research-only Parkinson's voice model training utilities.

Two supported data sources:
1. UCI Oxford telemonitoring CSV (has 'name' + 'status') - original loader kept.
2. UCI Parkinson Multiple Sound Recording files (user-supplied train_data.txt /
   test_data.txt, no headers). The loader assigns real column names, uses the
   first column as the participant group, and drops UPDRS (target leakage).

Hard rules enforced here:
- Participant-grouped cross-validation only. Recording-level random splits are
  rejected because repeated recordings of one participant leak information.
- test_data.txt is NOT used for evaluation: every row is labelled status=1 and
  its subjects overlap the train subjects, so no unbiased estimate is possible.
- No prediction is ever presented as a diagnosis.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

TARGET = "status"

MSR_COLUMNS = [
    "subject_id", "jitter_local", "jitter_local_abs", "jitter_rap", "jitter_ppq5",
    "jitter_ddp", "shimmer_local", "shimmer_local_db", "shimmer_apq3", "shimmer_apq5",
    "shimmer_apq11", "shimmer_dda", "ac", "nth", "htn", "median_pitch", "mean_pitch",
    "std_pitch", "min_pitch", "max_pitch", "num_pulses", "num_periods", "mean_period",
    "std_period", "frac_unvoiced", "num_voice_breaks", "degree_voice_breaks", "updrs", "status",
]
MSR_DROP = {"subject_id", "updrs", TARGET}


def load_multiple_sound_csv(path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load headerless multiple-sound-recording data with participant grouping."""
    df = pd.read_csv(path, header=None)
    n_cols = df.shape[1]
    if n_cols == len(MSR_COLUMNS):
        df.columns = MSR_COLUMNS
    elif n_cols == len(MSR_COLUMNS) - 1:
        df.columns = [c for c in MSR_COLUMNS if c != "updrs"]
    else:
        raise ValueError(f"Expected 28 or 29 columns, got {n_cols}.")
    if "status" not in df.columns:
        raise ValueError("No label column found (this file may be unlabelled).")
    y = pd.to_numeric(df["status"], errors="coerce")
    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError("status must contain only 0 (control) and 1 (Parkinson's label).")
    groups = df["subject_id"].astype(str).to_numpy()
    X = df.drop(columns=[c for c in MSR_DROP if c in df.columns]).apply(pd.to_numeric, errors="coerce")
    if X.shape[1] < 5:
        raise ValueError("Need at least five numeric voice features.")
    if len(np.unique(groups)) < 5:
        raise ValueError("Need at least five distinct participants.")
    return X, y.astype(int).to_numpy(), groups


def _subject_groups(names: pd.Series) -> np.ndarray:
    groups = []
    for raw in names.astype(str):
        match = re.search(r"(S\d+)", raw, flags=re.IGNORECASE)
        groups.append(match.group(1).upper() if match else re.sub(r"_\d+$", "", raw))
    return np.asarray(groups)


def load_uci_voice_csv(file_or_path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = file_or_path.copy() if isinstance(file_or_path, pd.DataFrame) else pd.read_csv(file_or_path)
    normalized = {str(c).strip().lower(): c for c in df.columns}
    if TARGET not in normalized or "name" not in normalized:
        raise ValueError("Expected the UCI Oxford format with 'name' and 'status' columns.")
    y = pd.to_numeric(df[normalized[TARGET]], errors="coerce")
    groups = _subject_groups(df[normalized["name"]])
    X = df.drop(columns=[normalized[TARGET], normalized["name"]]).apply(pd.to_numeric, errors="coerce")
    return X, y.astype(int).to_numpy(), groups


def _sel_lr():
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sel", SelectKBest(f_classif, k=8)),
                     ("sc", StandardScaler()),
                     ("clf", LogisticRegression(class_weight="balanced", max_iter=5000))])


def _sel_svm(random_state=42):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sel", SelectKBest(f_classif, k=8)),
                     ("sc", StandardScaler()),
                     ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=random_state))])


def _sel_rf(random_state=42):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sel", SelectKBest(f_classif, k=8)),
                     ("clf", RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                                                    random_state=random_state, max_features="sqrt"))])


def _sel_gb(random_state=42):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sel", SelectKBest(f_classif, k=8)),
                     ("clf", GradientBoostingClassifier(random_state=random_state))])


def _models(random_state=42):
    return {
        "LogisticRegression": _sel_lr(),
        "SVM_RBF": _sel_svm(random_state),
        "RandomForest": _sel_rf(random_state),
        "GradientBoosting": _sel_gb(random_state),
        "Ensemble(LR+GB)": VotingClassifier([("lr", _sel_lr()), ("gb", _sel_gb(random_state))], voting="soft"),
    }


def _fold_metrics(y_true, prob, groups_te):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    rec = dict(balanced_accuracy=balanced_accuracy_score(y_true, pred),
               sensitivity=tp/(tp+fn) if tp+fn else None,
               specificity=tn/(tn+fp) if tn+fp else None)
    yt, yp = [], []
    for g in np.unique(groups_te):
        m = groups_te == g
        yp.append(1 if prob[m].mean() >= 0.5 else 0)
        yt.append(y_true[m][0])
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    subj = dict(balanced_accuracy=((tp/(tp+fn))+(tn/(tn+fp)))/2 if (tp+fn and tn+fp) else None,
                sensitivity=tp/(tp+fn) if tp+fn else None,
                specificity=tn/(tn+fp) if tn+fp else None)
    return rec, subj


def train_parkinson_model(path, dataset="multiple_sound", n_splits=10, output_dir="models",
                          model_name="Ensemble(LR+GB)", random_state=42) -> dict:
    """Participant-grouped comparison of all models; fit selected model on all rows."""
    if dataset == "multiple_sound":
        X, y, groups = load_multiple_sound_csv(path)
    elif dataset == "oxford":
        X, y, groups = load_uci_voice_csv(path)
    else:
        raise ValueError("dataset must be 'multiple_sound' or 'oxford'.")

    class_groups = {int(l): len(np.unique(groups[y == l])) for l in (0, 1)}
    n_splits = min(n_splits, min(class_groups.values()))
    if n_splits < 3:
        raise ValueError("Need >=3 participants per class for grouped evaluation.")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    zoo = _models(random_state)
    comparison = {}
    for name, est in zoo.items():
        rec_acc, subj_acc = [], []
        for tr, te in cv.split(X, y, groups):
            est.fit(X.iloc[tr], y[tr])
            prob = est.predict_proba(X.iloc[te])[:, 1]
            rec, subj = _fold_metrics(y[te], prob, groups[te])
            rec_acc.append(rec["balanced_accuracy"]); subj_acc.append(subj["balanced_accuracy"])
        comparison[name] = {"recording_level_balanced_acc": float(np.mean(rec_acc)),
                            "subject_level_balanced_acc": float(np.mean(subj_acc))}

    selected = zoo[model_name]
    rec_log, subj_log = [], []
    for tr, te in cv.split(X, y, groups):
        selected.fit(X.iloc[tr], y[tr])
        prob = selected.predict_proba(X.iloc[te])[:, 1]
        rec, subj = _fold_metrics(y[te], prob, groups[te])
        rec_log.append(rec); subj_log.append(subj)

    # Out-of-fold predictions of the selected model: exact confusion matrix + ROC-AUC
    oof_prob = cross_val_predict(selected, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
    oof_pred = (oof_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, oof_pred, labels=[0, 1]).ravel()
    oof = {
        "balanced_accuracy": float(balanced_accuracy_score(y, oof_pred)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(precision_score(y, oof_pred, zero_division=0)),
        "f1": float(f1_score(y, oof_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, oof_prob)),
        "confusion_matrix": {"true_control": int(tn), "false_positive": int(fp),
                             "false_negative": int(fn), "true_parkinson_label": int(tp)},
    }

    def _agg(logs, key):
        vals = [d[key] for d in logs if d[key] is not None]
        return float(np.mean(vals))

    def _ci(vals_key):
        vals = [d[vals_key] for d in subj_log if d[vals_key] is not None]
        import scipy.stats as st
        lo, hi = st.t.interval(0.95, len(vals)-1, loc=np.mean(vals), scale=st.sem(vals))
        return [float(lo), float(hi)]

    metrics = {
        "recording_level": {k: _agg(rec_log, k) for k in ("balanced_accuracy", "sensitivity", "specificity")},
        "subject_level_mean_vote": {k: _agg(subj_log, k) for k in ("balanced_accuracy", "sensitivity", "specificity")},
        "subject_level_balanced_acc_95ci": _ci("balanced_accuracy"),
        "evaluation": f"{n_splits}-fold stratified grouped CV; participant recordings kept in one fold",
        "models_compared": comparison,
        "selected_model": model_name,
        "balanced_accuracy": oof["balanced_accuracy"],
        "sensitivity": oof["sensitivity"],
        "specificity": oof["specificity"],
        "precision": oof["precision"],
        "f1": oof["f1"],
        "roc_auc": oof["roc_auc"],
        "confusion_matrix": oof["confusion_matrix"],
    }

    selected.fit(X, y)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    model_path = out / "parkinson_voice_research.joblib"
    meta_path = out / "parkinson_voice_research_metadata.json"
    joblib.dump({"pipeline": selected, "feature_columns": list(X.columns),
                 "group_column": "subject_id", "threshold": 0.5,
                 "model_type": model_name}, model_path)
    metadata = {
        "model_name": "Parkinson's voice research model",
        "version": "2.0.0",
        "status": "experimental_not_for_patient_predictions",
        "dataset_rows": int(len(X)), "participant_count": int(len(np.unique(groups))),
        "feature_count": int(X.shape[1]),
        "dataset": {"source": dataset, "rows": int(len(X)),
                    "participant_count": int(len(np.unique(groups))),
                    "class_participant_counts": {"control": class_groups[0], "parkinsons_label": class_groups[1]},
                    "feature_count": int(X.shape[1]), "features": list(X.columns)},
        "metrics": metrics,
        "limitations": [
            "Participant-level signal is modest; some published scores for this dataset come from participant leakage (random splits).",
            "Grouped CV reduces leakage but is not external validation.",
            "Inputs are precomputed acoustic features, not NeuroNexus live MFCC summaries.",
            "Not validated for diagnosis, screening, treatment, or clinical decisions."],
        "model_file": str(model_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def train_parkinson_voice_model(file_or_path, output_dir="models"):
    """Backward-compatible alias for the NeuroNexus 2.0 research lab page.
    Sniffs the format: header with name+status columns = Oxford; otherwise
    headerless multiple-sound recording format."""
    df = file_or_path.copy() if isinstance(file_or_path, pd.DataFrame) else pd.read_csv(file_or_path, header=None, nrows=5)
    normalized = {str(c).strip().lower() for c in df.columns}
    dataset = "oxford" if {"name", "status"}.issubset(normalized) else "multiple_sound"
    return train_parkinson_model(file_or_path, dataset=dataset, output_dir=output_dir)
