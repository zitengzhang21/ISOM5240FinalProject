from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import (
    DATASET_NAME,
    EVALUATION_METRICS_FILE,
    EXPERIMENTAL_RESULTS_FILE,
    TEST_PREDICTIONS_FILE,
    TRAINING_METRICS_FILE,
)
from src.utils import ensure_directories


def read_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    if not TEST_PREDICTIONS_FILE.exists():
        raise FileNotFoundError("results/test_predictions.csv is missing. Run scripts/evaluate_model.py first.")

    ensure_directories([EXPERIMENTAL_RESULTS_FILE.parent])

    training_metrics = read_json(TRAINING_METRICS_FILE)
    evaluation_metrics = read_json(EVALUATION_METRICS_FILE)
    predictions_df = pd.read_csv(TEST_PREDICTIONS_FILE)

    model_selection_df = pd.DataFrame(
        [
            {
                "Pipeline": "Pipeline 1",
                "Model": "distilbert/distilbert-base-uncased (fine-tuned locally)",
                "Dataset": DATASET_NAME,
                "Accuracy": evaluation_metrics.get("accuracy"),
                "Precision": evaluation_metrics.get("precision"),
                "Recall": evaluation_metrics.get("recall"),
                "F1": evaluation_metrics.get("f1"),
                "Runtime": evaluation_metrics.get("average_inference_runtime_seconds"),
                "Decision": "Selected for phishing email detection",
            },
            {
                "Pipeline": "Pipeline 2",
                "Model": "facebook/bart-large-mnli (zero-shot classification)",
                "Dataset": DATASET_NAME,
                "Accuracy": None,
                "Precision": None,
                "Recall": None,
                "F1": None,
                "Runtime": "On-demand inference in Streamlit app",
                "Decision": "Selected for risk reason classification",
            },
        ]
    )

    app_performance_df = pd.DataFrame(
        [
            {"Metric": "Test Accuracy", "Value": evaluation_metrics.get("accuracy")},
            {"Metric": "Test Precision", "Value": evaluation_metrics.get("precision")},
            {"Metric": "Test Recall", "Value": evaluation_metrics.get("recall")},
            {"Metric": "Test F1", "Value": evaluation_metrics.get("f1")},
            {
                "Metric": "Average Inference Runtime (seconds)",
                "Value": evaluation_metrics.get("average_inference_runtime_seconds"),
            },
            {
                "Metric": "Training Device",
                "Value": training_metrics.get("device", "Not recorded"),
            },
        ]
    )

    test_predictions_sheet = predictions_df.rename(
        columns={
            "text": "sample text",
            "label": "true label",
            "predicted_label_name": "predicted label",
            "correct_or_incorrect": "correct/incorrect",
        }
    )
    keep_columns = [
        "sample text",
        "true label",
        "predicted label",
        "confidence",
        "correct/incorrect",
    ]
    test_predictions_sheet = test_predictions_sheet[keep_columns]

    with pd.ExcelWriter(EXPERIMENTAL_RESULTS_FILE, engine="openpyxl") as writer:
        model_selection_df.to_excel(writer, sheet_name="Model_Selection", index=False)
        app_performance_df.to_excel(writer, sheet_name="App_Performance", index=False)
        test_predictions_sheet.to_excel(writer, sheet_name="Test_Predictions", index=False)

    print(f"Saved experimental workbook to {EXPERIMENTAL_RESULTS_FILE}")


if __name__ == "__main__":
    main()
