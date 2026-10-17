import re
import joblib
from nltk.corpus import stopwords

# Load the trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Same cleaning as Step 3. It must match exactly.
stop_words = set(stopwords.words("english")) - {"not", "no", "nor", "never"}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

def predict_sentiment(text):
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prob_positive = model.predict_proba(features)[0][1]

    if prob_positive >= 0.60:
        label = "Positive"
    elif prob_positive <= 0.40:
        label = "Negative"
    else:
        label = "Neutral"
    return label, prob_positive

print("Sentiment Analyzer (type 'quit' to exit)")
while True:
    text = input("\nEnter a sentence: ")
    if text.lower() == "quit":
        break
    label, prob = predict_sentiment(text)
    print(f"Sentiment: {label}  (positive probability: {prob:.2f})")