import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")

# --- Model 1: our own trained model ---
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
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

# --- Model 2: pretrained VADER ---
vader = SentimentIntensityAnalyzer()

def vader_model(text):
    score = vader.polarity_scores(text)["compound"]   # from -1 to +1
    if score >= 0.05:
        return "Positive", score
    elif score <= -0.05:
        return "Negative", score
    return "Neutral", score

# --- Compare both on every input ---
print("Sentiment Comparison (type 'quit' to exit)")
while True:
    text = input("\nEnter a sentence: ")
    if text.lower() == "quit":
        break
    label1, p = our_model(text)
    label2, s = vader_model(text)
    print(f"Our model : {label1:<9} (positive probability: {p:.2f})")
    print(f"VADER     : {label2:<9} (score: {s:+.2f})")