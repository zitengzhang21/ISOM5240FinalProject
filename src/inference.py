from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from src.config import (
    FALLBACK_MODEL_NAME,
    LOCAL_MODEL_DIR,
    MAX_SEQUENCE_LENGTH,
    RISK_ACTIONS,
    RISK_REASON_LABELS,
    ZERO_SHOT_MODEL_NAME,
)
from src.utils import normalize_text


def _pipeline_device() -> int:
    return 0 if torch.cuda.is_available() else -1


def _normalize_prediction_label(raw_label: str) -> str:
    normalized = str(raw_label).strip().lower()
    mapping = {
        "label_0": "legitimate_email",
        "label_1": "phishing_email",
        "0": "legitimate_email",
        "1": "phishing_email",
        "legitimate_email": "legitimate_email",
        "phishing_email": "phishing_email",
        "safe": "legitimate_email",
        "legitimate": "legitimate_email",
        "phishing": "phishing_email",
    }
    return mapping.get(normalized, normalized)


def _normalize_classifier_output(raw_output):
    if isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], list):
        return raw_output[0]
    return raw_output


@lru_cache(maxsize=4)
def load_phishing_classifier(model_path: str = str(LOCAL_MODEL_DIR)):
    source = str(Path(model_path)) if Path(model_path).exists() else model_path
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source)
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        device=_pipeline_device(),
        top_k=None,
    )


@lru_cache(maxsize=1)
def load_zero_shot_classifier():
    return pipeline(
        "zero-shot-classification",
        model=ZERO_SHOT_MODEL_NAME,
        device=_pipeline_device(),
    )


def classify_risk_reason(
    text: str,
    classifier=None,
    prediction: Optional[str] = None,
) -> str:
    normalized_text = normalize_text(text)
    if not normalized_text:
        return "safe business email"

    if prediction == "legitimate_email":
        return "safe business email"

    zero_shot_classifier = classifier or load_zero_shot_classifier()
    candidate_labels = [label for label in RISK_REASON_LABELS if label != "safe business email"]
    result = zero_shot_classifier(
        normalized_text,
        candidate_labels=candidate_labels,
        hypothesis_template="This email is about {}.",
    )
    return str(result["labels"][0])


def get_risk_level(prediction: str, confidence: float) -> str:
    if prediction == "legitimate_email":
        return "Low"
    if confidence >= 0.85:
        return "High"
    if confidence >= 0.65:
        return "Medium"
    return "Low / Suspicious"


def get_suggested_action(prediction: str, risk_reason: str) -> str:
    if prediction == "legitimate_email":
        return RISK_ACTIONS["safe business email"]
    return RISK_ACTIONS.get(
        risk_reason,
        "Do not click any links. Report this email to the IT security team.",
    )


def get_outlook_label(prediction: str, confidence: float) -> str:
    if prediction == "legitimate_email":
        return "Safe Business Email"
    if confidence >= 0.85:
        return "Phishing Risk"
    if confidence >= 0.65:
        return "Suspicious Email"
    return "Needs Review"


def predict_phishing(
    text: str,
    classifier=None,
    zero_shot_classifier=None,
) -> dict:
    normalized_text = normalize_text(text)
    if not normalized_text:
        raise ValueError("Email text cannot be empty.")

    phishing_classifier = classifier or load_phishing_classifier()
    prediction_scores = _normalize_classifier_output(phishing_classifier(normalized_text))
    top_prediction = max(prediction_scores, key=lambda item: item["score"])

    prediction = _normalize_prediction_label(top_prediction["label"])
    confidence = float(top_prediction["score"])
    risk_reason = classify_risk_reason(
        normalized_text,
        classifier=zero_shot_classifier,
        prediction=prediction,
    )

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "risk_level": get_risk_level(prediction, confidence),
        "risk_reason": risk_reason,
        "suggested_action": get_suggested_action(prediction, risk_reason),
        "simulated_outlook_label": get_outlook_label(prediction, confidence),
    }


def default_model_source() -> str:
    return str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else FALLBACK_MODEL_NAME
