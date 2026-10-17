import re
import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("stopwords", quiet=True)
nltk.download("vader_lexicon", quiet=True)

# Load everything once and reuse it
@st.cache_resource
def load_models():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    vader = SentimentIntensityAnalyzer()
    return model, vectorizer, vader

model, vectorizer, vader = load_models()
stop_words = set(stopwords.words("english")) - {"not", "no", "nor", "never"}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

def our_model(text):
    features = vectorizer.transform([clean_text(text)])
    prob = model.predict_proba(features)[0][1]
    if prob >= 0.60:
        return "Positive", prob
    elif prob <= 0.40:
        return "Negative", prob
    return "Neutral", prob

def vader_model(text):
    score = vader.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive", score
    elif score <= -0.05:
        return "Negative", score
    return "Neutral", score

# ---------- The page ----------
st.title("Sentiment Analyzer")
st.write("Type a sentence and compare our trained model with the pretrained VADER model.")

text = st.text_area("Enter your text:", height=120)

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please type something first.")
    else:
        label1, prob = our_model(text)
        label2, score = vader_model(text)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Our model")
            st.metric("Sentiment", label1)
            st.write(f"Positive probability: {prob:.2f}")
            st.progress(float(prob))

        with col2:
            st.subheader("VADER (pretrained)")
            st.metric("Sentiment", label2)
            st.write(f"Compound score: {score:+.2f}")
            st.progress(float((score + 1) / 2))

        if label1 != label2:
            st.info("The two models disagree on this sentence.")