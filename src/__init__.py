"""Utilities for the phishing detection project."""

from .config import (
    BASE_MODEL_NAME,
    DATASET_NAME,
    FALLBACK_MODEL_NAME,
    LABEL_ID_TO_NAME,
    LABEL_NAME_TO_ID,
    ZERO_SHOT_MODEL_NAME,
)

__all__ = [
    "BASE_MODEL_NAME",
    "DATASET_NAME",
    "FALLBACK_MODEL_NAME",
    "LABEL_ID_TO_NAME",
    "LABEL_NAME_TO_ID",
    "ZERO_SHOT_MODEL_NAME",
]
