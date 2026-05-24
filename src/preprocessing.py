from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from datasets import DatasetDict, concatenate_datasets, load_dataset
from sklearn.model_selection import train_test_split

from src.config import DATASET_NAME, DEFAULT_RANDOM_STATE, MIN_TEXT_LENGTH


def load_phishing_dataset(dataset_name: str = DATASET_NAME):
    dataset = load_dataset(dataset_name)
    if isinstance(dataset, DatasetDict):
        splits = [dataset[split_name] for split_name in dataset.keys()]
        return concatenate_datasets(splits) if len(splits) > 1 else splits[0]
    return dataset


def clean_email_dataframe(
    dataframe: pd.DataFrame,
    min_text_length: int = MIN_TEXT_LENGTH,
) -> pd.DataFrame:
    df = dataframe.copy()

    rename_map = {}
    if "content" in df.columns:
        rename_map["content"] = "text"
    if "labels" in df.columns:
        rename_map["labels"] = "label"
    if rename_map:
        df = df.rename(columns=rename_map)

    required_columns = {"text", "label"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = df[df["label"].isin([0, 1])].copy()
    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset=["text"])
    df = df[df["text"].str.len() >= min_text_length]
    df["label"] = df["label"].astype(int)

    return df.reset_index(drop=True)


def create_balanced_sample(
    dataframe: pd.DataFrame,
    samples_per_class: int,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    sampled_frames = []

    for label, group in dataframe.groupby("label"):
        sample_size = min(samples_per_class, len(group))
        sampled_frames.append(group.sample(n=sample_size, random_state=random_state))

    balanced_df = pd.concat(sampled_frames, ignore_index=True)
    return balanced_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def stratified_split(
    dataframe: pd.DataFrame,
    train_size: float = 0.8,
    validation_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_size + validation_size + test_size, 1.0):
        raise ValueError("train_size + validation_size + test_size must equal 1.0")

    train_df, temp_df = train_test_split(
        dataframe,
        test_size=validation_size + test_size,
        stratify=dataframe["label"],
        random_state=random_state,
    )

    relative_test_size = test_size / (validation_size + test_size)
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df["label"],
        random_state=random_state,
    )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def label_distribution(dataframe: pd.DataFrame) -> dict:
    distribution = dataframe["label"].value_counts().sort_index().to_dict()
    return {int(label): int(count) for label, count in distribution.items()}


def print_label_distribution(split_name: str, dataframe: pd.DataFrame) -> None:
    distribution = label_distribution(dataframe)
    print(f"{split_name} label distribution: {distribution}")
