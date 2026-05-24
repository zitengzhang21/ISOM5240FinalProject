# %%
from pathlib import Path

import pandas as pd

from src.config import DATASET_NAME, PROCESSED_DATA_DIR
from src.preprocessing import (
    clean_email_dataframe,
    create_balanced_sample,
    load_phishing_dataset,
    stratified_split,
)

# %%
raw_dataset = load_phishing_dataset(DATASET_NAME)
raw_df = raw_dataset.to_pandas()
raw_df.head()

# %%
clean_df = clean_email_dataframe(raw_df)
clean_df["label"].value_counts()

# %%
balanced_df = create_balanced_sample(clean_df, samples_per_class=500)
balanced_df["label"].value_counts()

# %%
train_df, validation_df, test_df = stratified_split(balanced_df)
print(len(train_df), len(validation_df), len(test_df))

# %%
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
validation_df.to_csv(PROCESSED_DATA_DIR / "validation.csv", index=False)
test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

# %%
train_df.head()
