import re
import nltk
from datasets import load_dataset
from nltk.corpus import stopwords
import pandas as pd

nltk.download("stopwords")

# Stop words = very common words (the, is, and) that carry no sentiment.
# We keep negations, because "not good" is very different from "good".
stop_words = set(stopwords.words("english")) - {"not", "no", "nor", "never"}

def clean_text(text):
    text = text.lower()                       # 1. everything lowercase
    text = re.sub(r"<.*?>", " ", text)        # 2. remove HTML tags like <br />
    text = re.sub(r"n't\b", " not", text)     # 3. turn "didn't" into "did not"
    text = re.sub(r"[^a-z\s]", " ", text)     # 4. keep only letters and spaces
    words = text.split()                      # 5. split into a list of words
    words = [w for w in words if w not in stop_words]  # 6. drop stop words
    return " ".join(words)                    # 7. join back into one string

dataset = load_dataset("stanfordnlp/imdb")
train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

train_df["clean"] = train_df["text"].apply(clean_text)
test_df["clean"] = test_df["text"].apply(clean_text)

print("BEFORE:", train_df["text"][1][:200])
print("AFTER: ", train_df["clean"][1][:200])

# Save so we don't have to redo the cleaning every time
train_df.to_csv("clean_train.csv", index=False)
test_df.to_csv("clean_test.csv", index=False)