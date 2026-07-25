import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model and vectorizer
model = joblib.load('models/logistic_regression.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')

# Streamlit UI
st.title("📩 SMS & Email Spam Detector")
# ... rest of your code ...