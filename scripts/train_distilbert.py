from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.config import (
    BASE_MODEL_NAME,
    LABEL_ID_TO_NAME,
    LABEL_NAME_TO_ID,
    LOCAL_MODEL_DIR,
    MAX_SEQUENCE_LENGTH,
    RESULTS_DIR,
    TRAINING_METRICS_FILE,
    TRAIN_FILE,
    VALIDATION_FILE,
)
from src.utils import ensure_directories, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for phishing email classification.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    return parser.parse_args()


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    accuracy = accuracy_score(labels, predictions)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def load_split(path) -> Dataset:
    dataframe = pd.read_csv(path)
    return Dataset.from_pandas(dataframe, preserve_index=False)


def build_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    ensure_directories([RESULTS_DIR, LOCAL_MODEL_DIR.parent])
    kwargs = {
        "output_dir": str(RESULTS_DIR / "training_runs"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "report_to": [],
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "save_total_limit": 1,
        "fp16": torch.cuda.is_available(),
        "seed": 42,
    }

    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in signature:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        kwargs["eval_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


def build_trainer(
    model,
    training_args: TrainingArguments,
    train_dataset,
    validation_dataset,
    tokenizer,
):
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }

    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer

    return Trainer(**trainer_kwargs)


def main() -> None:
    args = parse_args()
    ensure_directories([LOCAL_MODEL_DIR.parent, RESULTS_DIR])

    if not TRAIN_FILE.exists() or not VALIDATION_FILE.exists():
        raise FileNotFoundError("Processed train/validation files are missing. Run scripts/prepare_data.py first.")

    train_dataset = load_split(TRAIN_FILE)
    validation_dataset = load_split(VALIDATION_FILE)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=2,
        id2label=LABEL_ID_TO_NAME,
        label2id=LABEL_NAME_TO_ID,
    )

    def tokenize_batch(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
        )

    train_dataset = train_dataset.map(tokenize_batch, batched=True)
    validation_dataset = validation_dataset.map(tokenize_batch, batched=True)

    columns = ["input_ids", "attention_mask", "label"]
    train_dataset.set_format(type="torch", columns=columns)
    validation_dataset.set_format(type="torch", columns=columns)

    trainer = build_trainer(
        model=model,
        training_args=build_training_arguments(args),
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        tokenizer=tokenizer,
    )

    train_result = trainer.train()
    evaluation_metrics = trainer.evaluate()

    trainer.save_model(str(LOCAL_MODEL_DIR))
    tokenizer.save_pretrained(str(LOCAL_MODEL_DIR))

    training_metrics = {
        "train_runtime": train_result.metrics.get("train_runtime"),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        "train_steps_per_second": train_result.metrics.get("train_steps_per_second"),
        "eval_accuracy": evaluation_metrics.get("eval_accuracy"),
        "eval_precision": evaluation_metrics.get("eval_precision"),
        "eval_recall": evaluation_metrics.get("eval_recall"),
        "eval_f1": evaluation_metrics.get("eval_f1"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_output_dir": str(LOCAL_MODEL_DIR),
    }
    save_json(training_metrics, TRAINING_METRICS_FILE)

    print(f"Saved trained model to {LOCAL_MODEL_DIR}")
    print(f"Saved training metrics to {TRAINING_METRICS_FILE}")


if __name__ == "__main__":
    main()
