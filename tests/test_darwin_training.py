import json

import numpy as np
import pandas as pd

from modules import alzheimers_model


def _darwin_frame(rows=30):
    rng = np.random.default_rng(42)
    labels = np.array(["H", "P"] * (rows // 2))
    values = rng.normal(size=(rows, 450))
    values[:, 0] += (labels == "P") * 0.8
    data = {f"feature_{index}": values[:, index] for index in range(450)}
    data["ID"] = [f"participant-{index:03d}" for index in range(rows)]
    data["class"] = labels
    return pd.DataFrame(data)


def test_darwin_loader_excludes_id_and_maps_labels(tmp_path):
    path = tmp_path / "darwin_data.csv"
    _darwin_frame(12).to_csv(path, index=False)

    features, labels, groups, columns = alzheimers_model.load_darwin_dataset(path)

    assert features.shape == (12, 450)
    assert len(columns) == 450
    assert "ID" not in columns and "class" not in columns
    assert set(labels) == {0, 1}
    assert len(set(groups)) == 12


def test_darwin_trainer_writes_calibrated_research_artifact_and_metrics(tmp_path):
    dataset_path = tmp_path / "darwin_data.csv"
    _darwin_frame().to_csv(dataset_path, index=False)
    output_dir = tmp_path / "models"

    metadata = alzheimers_model.train_darwin_model(
        dataset_path,
        output_dir=output_dir,
        n_splits=3,
        top_k=8,
        calibration_cv=2,
    )

    assert metadata["status"] == "experimental_research_only_no_patient_predictions"
    assert metadata["dataset"]["participant_count"] == 30
    assert metadata["dataset"]["license"] == "CC BY 4.0"
    assert len(metadata["feature_selection"]["selected_features"]) == 8
    assert metadata["preprocessing"]["scaling"] == "StandardScaler"
    assert metadata["preprocessing"]["calibration"]["method"] == "sigmoid"
    metrics = metadata["metrics"]["participant_level"]
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
    assert metadata["metrics"]["accuracy"] == metrics["accuracy"]
    assert (output_dir / "alzheimers_darwin_research.joblib").is_file()
    saved = json.loads((output_dir / "alzheimers_darwin_research_metadata.json").read_text())
    assert saved["metrics"]["participant_level"] == metrics


def test_registry_status_and_sample_profile_use_metadata_without_inference(tmp_path, monkeypatch):
    model_path = tmp_path / "alzheimers_darwin_research.joblib"
    metadata_path = tmp_path / "alzheimers_darwin_research_metadata.json"
    model_path.write_bytes(b"research artifact placeholder")
    metadata_path.write_text(json.dumps({
        "model_name": "DARWIN research model",
        "version": "1.0.0",
        "dataset": {"source": "uci_darwin", "participant_count": 174,
                    "cohort_median_profile": {"air_time1": 420.0}},
        "metrics": {"participant_level": {"accuracy": 0.61}},
        "validation_stage": 2,
    }))
    monkeypatch.setattr(alzheimers_model, "DARWIN_MODEL_PATH", model_path)
    monkeypatch.setattr(alzheimers_model, "DARWIN_METADATA_PATH", metadata_path)

    status = alzheimers_model.get_alzheimers_status()
    profile = alzheimers_model.load_darwin_sample_profile()

    assert status["available"] is True
    assert status["subject_level"]["accuracy"] == 0.61
    assert profile["features"] == {"air_time1": 420.0}
    assert "not an individual" in profile["notice"]
