"""Streamlit interface for the SMS Spam Detection project.

Run from the project root with: streamlit run app.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json


PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
SPAM_THRESHOLD = 0.5


def ensure_nltk_resources() -> None:
    """Download only the NLTK data required by the SVM preprocessing pipeline."""
    import nltk

    resources = {
        "tokenizers/punkt": "punkt",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for resource_path, package in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package, quiet=True)

    # Newer NLTK releases separate this resource from punkt.
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


@st.cache_resource(show_spinner="Loading trained models...")
def load_artifacts():
    """Load exactly the vectorizer, tokenizer, configuration, and models used in training."""
    ensure_nltk_resources()

    svm_model = joblib.load(MODELS_DIR / "svm_model.pkl")
    tfidf_vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
    gru_model = tf.keras.models.load_model(MODELS_DIR / "gru_spam_model.keras")

    tokenizer_json = (MODELS_DIR / "gru_tokenizer.json").read_text(encoding="utf-8")
    gru_tokenizer = tokenizer_from_json(tokenizer_json)
    gru_config = json.loads((MODELS_DIR / "gru_sequence_config.json").read_text(encoding="utf-8"))

    return svm_model, tfidf_vectorizer, gru_model, gru_tokenizer, gru_config


def preprocess_for_svm(message: str) -> str:
    """Replicate the preprocessing in notebooks/03_text_preprocessing.ipynb."""
    text = str(message).lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\S+@\S+\.\S+", " email ", text)
    text = re.sub(r"[$£€₹]", " money ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    return " ".join(
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 1
    )


def preprocess_for_gru(message: str) -> str:
    """Replicate the lighter GRU preprocessing used during tokenizer training."""
    text = str(message).lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\S+@\S+\.\S+", " email ", text)
    text = re.sub(r"[$£€₹]", " money ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def predict_svm(message: str, svm_model, vectorizer) -> tuple[str, float]:
    cleaned_message = preprocess_for_svm(message)
    features = vectorizer.transform([cleaned_message])
    predicted_class = int(svm_model.predict(features)[0])

    if hasattr(svm_model, "predict_proba"):
        spam_probability = float(svm_model.predict_proba(features)[0][1])
    else:
        # The saved LinearSVC has no calibrated probabilities; report a score-derived confidence.
        decision = float(svm_model.decision_function(features)[0])
        spam_probability = 1 / (1 + np.exp(-decision))

    confidence = spam_probability if predicted_class == 1 else 1 - spam_probability
    return ("Spam" if predicted_class == 1 else "Ham"), confidence


def predict_gru(message: str, model, tokenizer, config: dict) -> tuple[str, float]:
    cleaned_message = preprocess_for_gru(message)
    sequence = tokenizer.texts_to_sequences([cleaned_message])
    padded_sequence = pad_sequences(
        sequence,
        maxlen=int(config["max_length"]),
        padding=config["padding"],
        truncating=config["truncating"],
    )
    spam_probability = float(model.predict(padded_sequence, verbose=0)[0][0])
    label = "Spam" if spam_probability >= SPAM_THRESHOLD else "Ham"
    confidence = spam_probability if label == "Spam" else 1 - spam_probability
    return label, confidence


def show_prediction(model_name: str, prediction: tuple[str, float]) -> None:
    label, confidence = prediction
    if label == "Spam":
        st.error(f"{model_name}: SPAM")
    else:
        st.success(f"{model_name}: HAM")
    st.metric("Confidence", f"{confidence:.1%}")


def show_model_comparison() -> None:
    comparison_path = RESULTS_DIR / "model_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
    else:
        comparison = pd.DataFrame(
            {
                "Model": ["SVM", "GRU"],
                "Accuracy": [0.9841, 0.9851],
                "Precision": [0.9808, 0.9244],
                "Recall": [0.8793, 0.9483],
                "F1_Score": [0.9273, 0.9362],
                "ROC_AUC": [0.9943, 0.9958],
            }
        )

    st.subheader("Model performance")
    st.dataframe(
        comparison.set_index("Model").style.format("{:.2%}"),
        use_container_width=True,
    )
    st.bar_chart(comparison.set_index("Model"))


def main() -> None:
    st.set_page_config(page_title="SMS Spam Detector", page_icon="📱", layout="centered")
    st.title("📱 SMS Spam Detector")
    st.caption("Classify SMS messages using your trained SVM and GRU models.")

    page = st.sidebar.radio("Navigate", ["Detector", "Model comparison", "About"])

    if page == "Model comparison":
        show_model_comparison()
        return

    if page == "About":
        st.subheader("About this application")
        st.write(
            "The SVM uses TF-IDF features, while the GRU uses token sequences. "
            "Both use the exact preprocessing and saved artifacts from this project."
        )
        st.info("Predictions are decision support only; review suspicious messages before acting.")
        return

    try:
        svm_model, vectorizer, gru_model, gru_tokenizer, gru_config = load_artifacts()
    except Exception as exc:
        st.error("The trained model files could not be loaded.")
        st.exception(exc)
        return

    message = st.text_area(
        "Enter an SMS message",
        placeholder="Example: Congratulations! You have won a free prize. Reply now to claim.",
        height=140,
    )
    selected_model = st.radio("Choose a model", ["GRU", "SVM", "Compare both"], horizontal=True)

    if st.button("Detect SMS", type="primary"):
        if not message.strip():
            st.warning("Please enter an SMS message first.")
            return

        if selected_model == "SVM":
            show_prediction("SVM", predict_svm(message, svm_model, vectorizer))
        elif selected_model == "GRU":
            show_prediction("GRU", predict_gru(message, gru_model, gru_tokenizer, gru_config))
        else:
            left, right = st.columns(2)
            with left:
                show_prediction("SVM", predict_svm(message, svm_model, vectorizer))
            with right:
                show_prediction("GRU", predict_gru(message, gru_model, gru_tokenizer, gru_config))


if __name__ == "__main__":
    main()
