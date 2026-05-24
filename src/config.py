from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
IMAGES_DIR = PROJECT_ROOT / "images"

TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
VALIDATION_FILE = PROCESSED_DATA_DIR / "validation.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"
APP_TEST_FILE = PROCESSED_DATA_DIR / "app_test_emails.csv"
SAMPLE_TEST_EMAILS_FILE = DATA_DIR / "sample_test_emails.csv"

LOCAL_MODEL_DIR = MODELS_DIR / "phishing-distilbert-local"
TRAINING_METRICS_FILE = RESULTS_DIR / "training_metrics.json"
EVALUATION_METRICS_FILE = RESULTS_DIR / "evaluation_metrics.json"
TEST_PREDICTIONS_FILE = RESULTS_DIR / "test_predictions.csv"
EXPERIMENTAL_RESULTS_FILE = RESULTS_DIR / "experimental_results.xlsx"

DATASET_NAME = "cybersectony/PhishingEmailDetectionv2.0"
BASE_MODEL_NAME = "distilbert/distilbert-base-uncased"
FALLBACK_MODEL_NAME = "cybersectony/phishing-email-detection-distilbert_v2.4.1"
ZERO_SHOT_MODEL_NAME = "facebook/bart-large-mnli"

LABEL_ID_TO_NAME = {
    0: "legitimate_email",
    1: "phishing_email",
}
LABEL_NAME_TO_ID = {value: key for key, value in LABEL_ID_TO_NAME.items()}

RISK_REASON_LABELS = [
    "credential theft",
    "fake payment request",
    "malware link",
    "account verification scam",
    "prize or reward scam",
    "safe business email",
]

RISK_ACTIONS = {
    "credential theft": "Do not enter credentials. Report to IT security.",
    "fake payment request": "Verify with the finance team before making any payment.",
    "malware link": "Do not open links or attachments.",
    "account verification scam": "Verify the request through the official website, not the email link.",
    "prize or reward scam": "Ignore the message and report it as suspicious.",
    "safe business email": "No immediate action required, but verify sender identity if unsure.",
}

DEFAULT_RANDOM_STATE = 42
MIN_TEXT_LENGTH = 20
MAX_SEQUENCE_LENGTH = 256
