from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    APP_TEST_FILE,
    DATASET_NAME,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from src.preprocessing import (
    clean_email_dataframe,
    create_balanced_sample,
    load_phishing_dataset,
    print_label_distribution,
    stratified_split,
)
from src.utils import ensure_directories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare phishing email datasets for local training.")
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=1000,
        help="Maximum number of examples to keep for each class in the local development dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories([RAW_DATA_DIR, PROCESSED_DATA_DIR])

    raw_dataset = load_phishing_dataset(DATASET_NAME)
    raw_dataframe = raw_dataset.to_pandas()

    cleaned_dataframe = clean_email_dataframe(raw_dataframe)
    balanced_dataframe = create_balanced_sample(
        cleaned_dataframe,
        samples_per_class=args.samples_per_class,
    )

    train_df, validation_df, test_df = stratified_split(balanced_dataframe)

    train_path = PROCESSED_DATA_DIR / "train.csv"
    validation_path = PROCESSED_DATA_DIR / "validation.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(validation_path, index=False)
    test_df.to_csv(test_path, index=False)

    app_test_df = test_df.sample(
        n=min(25, len(test_df)),
        random_state=42,
    ).reset_index(drop=True)
    app_test_df.to_csv(APP_TEST_FILE, index=False)

    print(f"Saved train split to {train_path}")
    print(f"Saved validation split to {validation_path}")
    print(f"Saved test split to {test_path}")
    print(f"Saved app test samples to {APP_TEST_FILE}")

    print_label_distribution("Train", train_df)
    print_label_distribution("Validation", validation_df)
    print_label_distribution("Test", test_df)


if __name__ == "__main__":
    main()
