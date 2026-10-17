import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the cleaned data from the last step
train_df = pd.read_csv("clean_train.csv")
test_df = pd.read_csv("clean_test.csv")

# A review that became empty after cleaning would load as NaN, so fix that
train_df["clean"] = train_df["clean"].fillna("")
test_df["clean"] = test_df["clean"].fillna("")

# TF-IDF: keep the 50,000 most useful words and word pairs
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_df["clean"])   # learn vocabulary + convert
X_test = vectorizer.transform(test_df["clean"])         # convert only
y_train = train_df["label"]
y_test = test_df["label"]

# Train the classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate on reviews the model has never seen
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

# Save both so we can reuse them without retraining
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(model, "model.pkl")