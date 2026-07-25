import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are available
nltk.download('stopwords')
nltk.download('wordnet')

# Load the best-performing model (Logistic Regression) & TF-IDF Vectorizer
@st.cache_resource
def load_assets():
    model = joblib.load('models/logistic_regression.pkl')
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    return model, tfidf

model, tfidf = load_assets()

# Preprocessing function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# Streamlit User Interface
st.set_page_config(page_title="SMS & Email Spam Detector", page_icon="📩", layout="centered")

st.title("📩 SMS & Email Spam Detector")
st.write("Group 31 - Natural Language Processing System")
st.markdown("---")

st.subheader("Message Classification")
user_input = st.text_area("Enter the email or SMS text to analyze:", height=150, placeholder="Paste message here...")

if st.button("Analyze Message", type="primary"):
    if user_input.strip() != "":
        # Preprocess & Vectorize
        cleaned_text = preprocess_text(user_input)
        vectorized_text = tfidf.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(vectorized_text)[0]
        confidence = model.predict_proba(vectorized_text)[0][prediction] * 100
        
        st.markdown("### Prediction Result:")
        if prediction == 1:
            st.error(f"🚨 **SPAM DETECTED**\n\n**Confidence Level:** {confidence:.2f}%")
        else:
            st.success(f"✅ **HAM (Legitimate Message)**\n\n**Confidence Level:** {confidence:.2f}%")
    else:
        st.warning("Please enter a message before clicking analyze.")

st.markdown("---")
st.caption("Sri Lanka Technology Campus | CCS3356 NLP Group Assignment")