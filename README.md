# AI-powered Email Spam and Phishing Detection System for Corporate Cybersecurity

## Project Overview
This project is a local-first Streamlit application for detecting whether an email is legitimate or phishing. The system is designed for a corporate cybersecurity use case and focuses on deep learning text classification rather than chatbot-style generation.

The app provides:
- prediction label
- confidence score
- risk level
- risk reason
- suggested action
- simulated Outlook label

## Business Objective
The main business objective is to help employees or security teams quickly triage suspicious emails and reduce phishing risk inside an organization. The project emphasizes explainable decision support with a classification-focused workflow that can later be deployed to Streamlit Cloud.

## Model Design
This project uses two Hugging Face pipelines:

1. Pipeline 1: Phishing email classification
- Base model: `distilbert/distilbert-base-uncased`
- Task: binary text classification
- Labels:
  - `0 = legitimate_email`
  - `1 = phishing_email`
- Approach: fine-tune locally on a balanced development sample

2. Pipeline 2: Risk reason classification
- Model: `facebook/bart-large-mnli`
- Task: zero-shot classification
- Candidate labels:
  - `credential theft`
  - `fake payment request`
  - `malware link`
  - `account verification scam`
  - `prize or reward scam`
  - `safe business email`

## Dataset
- Hugging Face dataset: `cybersectony/PhishingEmailDetectionv2.0`
- Text column: `content`
- Label column: `labels`

Preprocessing rules:
- keep only labels `0` and `1`
- ignore URL labels `2` and `3`
- rename `content` to `text`
- rename `labels` to `label`
- remove empty text
- remove duplicated text
- remove very short text under 20 characters
- create balanced local development samples

## Project Structure
```text
.
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- preprocessing.py
|   |-- inference.py
|   `-- utils.py
|-- scripts/
|   |-- prepare_data.py
|   |-- train_distilbert.py
|   |-- evaluate_model.py
|   `-- make_experimental_results.py
|-- notebooks/
|   |-- 01_data_preprocessing.py
|   |-- 02_finetune_distilbert.py
|   `-- 03_experiments_and_testing.py
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- sample_test_emails.csv
|-- models/
|   `-- .gitkeep
|-- results/
|   `-- .gitkeep
`-- images/
    `-- .gitkeep
```

## Local Setup
Create and activate a virtual environment, then install the dependencies.

```bash
python -m venv .venv
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

Install packages:

```bash
pip install -r requirements.txt
```

## Command Sequence
Run the project in this order:

```bash
python scripts/prepare_data.py --samples_per_class 500
python scripts/train_distilbert.py --epochs 1 --batch_size 8
python scripts/evaluate_model.py
python scripts/make_experimental_results.py
streamlit run app.py
```

## How to Prepare Data
Use the preprocessing script to download the dataset, clean it, balance it, split it, and save local CSV files.

```bash
python scripts/prepare_data.py --samples_per_class 500
```

Generated files:
- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`
- `data/processed/app_test_emails.csv`

## How to Train Locally
Fine-tune DistilBERT using the prepared train and validation files.

```bash
python scripts/train_distilbert.py --epochs 1 --batch_size 8
```

Outputs:
- `models/phishing-distilbert-local/`
- `results/training_metrics.json`

## How to Evaluate
Run the evaluation script on the test split to measure model quality and inference speed.

```bash
python scripts/evaluate_model.py
```

Outputs:
- `results/test_predictions.csv`
- `results/evaluation_metrics.json`

## How to Generate Experimental Results
Create an Excel workbook that summarizes the project experiment outputs.

```bash
python scripts/make_experimental_results.py
```

Output:
- `results/experimental_results.xlsx`

## How to Run the Streamlit App
Start the local web app after training and evaluation.

```bash
streamlit run app.py
```

App behavior:
- uses `st.cache_resource` to avoid reloading models on every interaction
- warns clearly if the local fine-tuned model is missing
- optionally falls back to `cybersectony/phishing-email-detection-distilbert_v2.4.1` for demo only
- clearly labels the fallback as a baseline fallback model
- does not use external LLM APIs

## Notebooks
The `notebooks/` folder contains Python percent-cell notebook files using `# %%` so they can be opened in VS Code or converted to `.ipynb` later.

## Deployment Notes
The project is designed to run locally first and can later be adapted for Streamlit Cloud deployment. Large generated model files, raw data dumps, and evaluation outputs should stay untracked.

## Placeholders
- GitHub repository: `https://github.com/zitengzhang21/ISOM5240FinalProject`
- Hugging Face model URL placeholder: `https://huggingface.co/<your-username>/<your-model-repo>`
- Streamlit Cloud app URL placeholder: `https://<your-streamlit-app-url>`

## What to Commit
Commit the source code, app, scripts, notebooks, README, requirements file, sample test CSV, and placeholder `.gitkeep` files. Do not commit large generated model artifacts or processed data outputs.
