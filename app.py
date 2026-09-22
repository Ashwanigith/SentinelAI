import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

st.set_page_config(
    page_title="SentinelAI - Spam & Threat Detection",
    page_icon="🛡️",
    layout="wide"
)

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

@st.cache_resource
def load_assets():
    model = joblib.load('model/spam_classifier.pkl')
    vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
    return model, vectorizer

try:
    model, vectorizer = load_assets()
except Exception:
    st.error("Model artifacts not found! Please run 'python train_model.py' first.")
    st.stop()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' urltoken ', text)
    text = re.sub(r'\d+', ' numtoken ', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

st.title("🛡️ SentinelAI")
st.markdown("**Intelligent Spam & Threat Detection System** powered by NLP & Machine Learning.")
st.write("---")

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area(
        "Analyze Incoming Text / Message:",
        placeholder="Paste your email, SMS, or notification content here...",
        height=180
    )
    scan_button = st.button("🔍 Scan for Threats", type="primary", use_container_width=True)

with col2:
    st.subheader("Inspection Indicators")
    has_url = bool(re.search(r'https?://\S+|www\.\S+', user_input)) if user_input else False
    has_urgency = bool(re.search(r'\b(urgent|warning|blocked|suspended|otp|verify|immediately)\b', user_input, re.I)) if user_input else False
    
    st.metric("Suspicious Link Detected", "Yes" if has_url else "No")
    st.metric("Urgency Language Marker", "Yes" if has_urgency else "No")

if scan_button and user_input.strip():
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    confidence = max(probabilities) * 100

    st.write("---")
    st.subheader("Analysis Verdict")
    
    if prediction == "Threat/Spam":
        st.error(f"⚠️ **Threat / Spam Detected!** (Confidence: {confidence:.2f}%)")
        st.warning("Recommendation: Do not click any links, share OTPs, or reply with personal data.")
    else:
        st.success(f"✅ **Safe (Ham) Content** (Confidence: {confidence:.2f}%)")
        st.info("No critical threat signatures or phishing patterns detected.")