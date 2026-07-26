import os
import joblib
import streamlit as st

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "logistic_regression.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

# Page Config
st.set_page_config(page_title="SMS & Email Spam Detector", page_icon="📩")

# Header
st.title("📩 SMS & Email Spam Detector")
st.caption("Group 31 - Natural Language Processing System")
st.write("---")

st.subheader("Message Classification")

# Input text box
input_text = st.text_area(
    "Enter the email or SMS text to analyze:",
    value="WINNER!! You have won a $1000 gift card! Call now to claim your prize!",
    height=120
)

# Button
if st.button("Analyze Message", type="primary"):
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        try:
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            
            transformed = vectorizer.transform([input_text])
            pred = model.predict(transformed)[0]
            probs = model.predict_proba(transformed)[0]
            
            st.write("### Prediction Result:")
            if pred == 1 or pred == "spam":
                st.error(f"🚨 **SPAM DETECTED**\n\n**Confidence Level:** {probs[1]*100:.2f}%")
            else:
                st.success(f"✅ **NOT SPAM (HAM)**\n\n**Confidence Level:** {probs[0]*100:.2f}%")
        except Exception as e:
            st.error(f"Error loading model: {e}")