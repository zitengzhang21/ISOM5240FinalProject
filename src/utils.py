import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def combine_subject_body(subject: object, body: object) -> str:
    subject_text = normalize_text(subject)
    body_text = normalize_text(body)
    parts = [part for part in (subject_text, body_text) if part]
    return "\n\n".join(parts)


def build_text_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()

    if "text" in df.columns:
        df["text"] = df["text"].map(normalize_text)
        return df

    if {"subject", "body"}.issubset(df.columns):
        df["text"] = df.apply(
            lambda row: combine_subject_body(row.get("subject"), row.get("body")),
            axis=1,
        )
        return df

    raise ValueError("CSV must contain either a 'text' column or both 'subject' and 'body' columns.")


def label_name_from_id(label_id: int) -> str:
    return "phishing_email" if int(label_id) == 1 else "legitimate_email"
