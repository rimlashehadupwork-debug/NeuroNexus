"""Train the internal grouped-CV DARWIN kinematic research artifact."""
from __future__ import annotations

import argparse

from modules.alzheimers_model import (
    DARWIN_TOP_K,
    DEFAULT_DARWIN_DATASET,
    MODELS_DIR,
    train_darwin_model,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DARWIN_DATASET),
        help="Path to data/darwin_data.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(MODELS_DIR),
        help="Directory for the model artifact and metadata JSON",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=DARWIN_TOP_K)
    args = parser.parse_args()

    metadata = train_darwin_model(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        n_splits=args.folds,
        top_k=args.top_k,
    )
    metrics = metadata["metrics"]["participant_level"]
    print(f"Dataset: {metadata['dataset']['rows']} rows / {metadata['participant_count']} participants")
    print(f"Features selected: {', '.join(metadata['feature_selection']['selected_features'])}")
    print(f"Grouped-CV accuracy: {metrics['accuracy']:.3f}")
    print(f"Grouped-CV ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"Research artifact: {metadata['model_file']}")
    print("Research only: no individual prediction or clinical risk estimate is produced.")


if __name__ == "__main__":
    main()
