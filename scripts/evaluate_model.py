from __future__ import annotations

import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.config import (
    EVALUATION_METRICS_FILE,
    LOCAL_MODEL_DIR,
    TEST_FILE,
    TEST_PREDICTIONS_FILE,
)
from src.inference import load_phishing_classifier
from src.utils import ensure_directories, save_json


def main() -> None:
    ensure_directories([TEST_PREDICTIONS_FILE.parent])

    if not LOCAL_MODEL_DIR.exists():
        raise FileNotFoundError("Local model is missing. Run scripts/train_distilbert.py first.")
    if not TEST_FILE.exists():
        raise FileNotFoundError("Processed test split is missing. Run scripts/prepare_data.py first.")

    test_df = pd.read_csv(TEST_FILE)
    classifier = load_phishing_classifier(str(LOCAL_MODEL_DIR))

    predictions = []
    runtimes = []

    for text in test_df["text"].astype(str):
        start_time = time.perf_counter()
        raw_scores = classifier(text)
        scores = raw_scores[0] if isinstance(raw_scores, list) and raw_scores and isinstance(raw_scores[0], list) else raw_scores
        elapsed = time.perf_counter() - start_time

        top_prediction = max(scores, key=lambda item: item["score"])
        predicted_label = 1 if "1" in str(top_prediction["label"]).lower() or "phishing" in str(top_prediction["label"]).lower() else 0
        confidence = float(top_prediction["score"])

        predictions.append((predicted_label, confidence))
        runtimes.append(elapsed)

    predicted_labels = [item[0] for item in predictions]
    confidences = [item[1] for item in predictions]
    true_labels = test_df["label"].astype(int).tolist()

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="binary",
        zero_division=0,
    )
    metrics = {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels).tolist(),
        "average_inference_runtime_seconds": sum(runtimes) / max(len(runtimes), 1),
        "model_path": str(Path(LOCAL_MODEL_DIR)),
    }

    output_df = test_df.copy()
    output_df["predicted_label"] = predicted_labels
    output_df["predicted_label_name"] = output_df["predicted_label"].map(
        {0: "legitimate_email", 1: "phishing_email"}
    )
    output_df["confidence"] = confidences
    output_df["correct_or_incorrect"] = output_df.apply(
        lambda row: "correct" if int(row["label"]) == int(row["predicted_label"]) else "incorrect",
        axis=1,
    )

    output_df.to_csv(TEST_PREDICTIONS_FILE, index=False)
    save_json(metrics, EVALUATION_METRICS_FILE)

    print(f"Saved test predictions to {TEST_PREDICTIONS_FILE}")
    print(f"Saved evaluation metrics to {EVALUATION_METRICS_FILE}")


if __name__ == "__main__":
    main()
