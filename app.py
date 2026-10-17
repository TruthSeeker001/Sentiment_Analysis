import re
import joblib
import nltk
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("stopwords", quiet=True)
nltk.download("vader_lexicon", quiet=True)

st.set_page_config(page_title="Sentiment Analyzer")

# ---------- Loading (cached so it happens once) ----------
@st.cache_resource
def load_text_models():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    vader = SentimentIntensityAnalyzer()
    return model, vectorizer, vader

@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

model, vectorizer, vader = load_text_models()
face_detector = load_face_detector()
stop_words = set(stopwords.words("english")) - {"not", "no", "nor", "never"}

# ---------- Text functions ----------
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

# ---------- Face function ----------
def analyze_face(image_bytes):
    from deepface import DeepFace   # imported here so the text tab starts fast

    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None, None, None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])   # the biggest face
    result = DeepFace.analyze(
        frame[y:y + h, x:x + w],
        actions=["emotion"],
        detector_backend="skip",
        enforce_detection=False,
        silent=True,
    )
    if isinstance(result, list):
        result = result[0]

    emotion = result["dominant_emotion"]
    scores = result["emotion"]

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(frame, f"{emotion} {scores[emotion]:.0f}%", (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return frame, emotion, scores

# ---------- The page ----------
st.title("Sentiment Analyzer")
tab_text, tab_face = st.tabs(["Text sentiment", "Face emotion"])

with tab_text:
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

with tab_face:
    st.write("Take a photo and the app will detect your face and guess the emotion (DeepFace, pretrained).")
    photo = st.camera_input("Take a photo")

    if photo is not None:
        with st.spinner("Analyzing..."):
            frame, emotion, scores = analyze_face(photo.getvalue())

        if frame is None:
            st.warning("No face found. Try better lighting and face the camera directly.")
        else:
            st.image(frame, channels="BGR")
            st.subheader(f"Detected emotion: {emotion}")
            st.bar_chart(pd.Series(scores))