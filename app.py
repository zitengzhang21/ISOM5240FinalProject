from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import (
    DATASET_NAME,
    EXPERIMENTAL_RESULTS_FILE,
    FALLBACK_MODEL_NAME,
    LABEL_ID_TO_NAME,
    LOCAL_MODEL_DIR,
)
from src.inference import load_phishing_classifier, load_zero_shot_classifier, predict_phishing
from src.utils import build_text_column


st.set_page_config(
    page_title="AI-powered Email Spam and Phishing Detection System",
    page_icon=":shield:",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_models(use_fallback: bool):
    model_source = FALLBACK_MODEL_NAME if use_fallback else str(LOCAL_MODEL_DIR)
    phishing_classifier = load_phishing_classifier(model_source)
    zero_shot_classifier = load_zero_shot_classifier()
    return phishing_classifier, zero_shot_classifier


def analyze_text(text: str, use_fallback: bool) -> dict:
    phishing_classifier, zero_shot_classifier = get_models(use_fallback)
    return predict_phishing(
        text,
        classifier=phishing_classifier,
        zero_shot_classifier=zero_shot_classifier,
    )


def render_prediction(result: dict) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Prediction Label", result["prediction"])
    col2.metric("Confidence Score", f'{result["confidence"]:.2%}')
    col3.metric("Risk Level", result["risk_level"])

    st.write(f"**Risk Reason:** {result['risk_reason']}")
    st.write(f"**Suggested Action:** {result['suggested_action']}")
    st.write(f"**Simulated Outlook Label:** {result['simulated_outlook_label']}")


def render_experiment_summary() -> None:
    if not EXPERIMENTAL_RESULTS_FILE.exists():
        st.info(
            "Experimental results are not available yet. Run `python scripts/evaluate_model.py` "
            "and `python scripts/make_experimental_results.py` after training."
        )
        return

    model_selection_df = pd.read_excel(EXPERIMENTAL_RESULTS_FILE, sheet_name="Model_Selection")
    app_performance_df = pd.read_excel(EXPERIMENTAL_RESULTS_FILE, sheet_name="App_Performance")

    st.dataframe(model_selection_df, use_container_width=True)
    st.dataframe(app_performance_df, use_container_width=True)


def main() -> None:
    st.title("AI-powered Email Spam and Phishing Detection System for Corporate Cybersecurity")
    st.write(
        "This Streamlit app classifies whether an email is legitimate or phishing, then "
        "adds a confidence score, risk level, risk reason, suggested action, and a simulated Outlook label."
    )
    st.caption("No external LLM APIs are used. The system relies on Hugging Face classification pipelines only.")

    local_model_exists = LOCAL_MODEL_DIR.exists()
    use_fallback = False

    with st.sidebar:
        st.header("Runtime Options")
        if local_model_exists:
            st.success(f"Local fine-tuned model detected at `{LOCAL_MODEL_DIR}`.")
        else:
            st.warning(
                "Local fine-tuned model not found. Training is required for the intended workflow."
            )
            use_fallback = st.checkbox(
                "Use baseline fallback model for demo only",
                value=True,
                help="Loads cybersectony/phishing-email-detection-distilbert_v2.4.1 as a temporary baseline.",
            )
            if use_fallback:
                st.info(f"Baseline fallback active: `{FALLBACK_MODEL_NAME}`")

    st.header("1. Single Email Detection")
    single_email_text = st.text_area(
        "Paste email content here",
        height=220,
        placeholder="Enter the full email body, or combine subject and body for richer analysis.",
    )

    if st.button("Analyze Email", type="primary"):
        if not single_email_text.strip():
            st.error("Please enter email content before running analysis.")
        elif not local_model_exists and not use_fallback:
            st.error("Local model is missing. Train the model first or enable the baseline fallback model.")
        else:
            with st.spinner("Analyzing email..."):
                result = analyze_text(single_email_text, use_fallback=use_fallback)
            render_prediction(result)

    st.header("2. Batch Email Detection")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            prepared_df = build_text_column(uploaded_df)
            prepared_df = prepared_df[prepared_df["text"].str.strip() != ""].reset_index(drop=True)

            if st.button("Run Batch Analysis"):
                if not local_model_exists and not use_fallback:
                    st.error("Local model is missing. Train the model first or enable the baseline fallback model.")
                else:
                    with st.spinner("Running batch analysis..."):
                        results = [
                            analyze_text(text, use_fallback=use_fallback)
                            for text in prepared_df["text"].astype(str)
                        ]
                    results_df = pd.concat([uploaded_df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
                    st.dataframe(results_df, use_container_width=True)
                    st.download_button(
                        "Download Predictions CSV",
                        data=results_df.to_csv(index=False).encode("utf-8"),
                        file_name="batch_email_predictions.csv",
                        mime="text/csv",
                    )
        except Exception as exc:
            st.error(f"Unable to process the uploaded CSV: {exc}")

    st.header("3. Model Information")
    model_info_df = pd.DataFrame(
        [
            {"Field": "Dataset", "Value": DATASET_NAME},
            {"Field": "Phishing Classifier", "Value": "distilbert/distilbert-base-uncased (fine-tuned locally)"},
            {"Field": "Risk Reason Classifier", "Value": "facebook/bart-large-mnli (zero-shot classification)"},
            {"Field": "Fallback Baseline", "Value": FALLBACK_MODEL_NAME},
            {"Field": "Label 0", "Value": LABEL_ID_TO_NAME[0]},
            {"Field": "Label 1", "Value": LABEL_ID_TO_NAME[1]},
            {"Field": "Project Purpose", "Value": "Corporate phishing detection and risk triage for email review."},
        ]
    )
    st.dataframe(model_info_df, use_container_width=True, hide_index=True)

    st.header("4. Experiment Results")
    render_experiment_summary()


if __name__ == "__main__":
    main()
