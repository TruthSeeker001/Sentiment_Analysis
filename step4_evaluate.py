import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Load the saved model, vectorizer, and test data
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
test_df = pd.read_csv("clean_test.csv")
test_df["clean"] = test_df["clean"].fillna("")

X_test = vectorizer.transform(test_df["clean"])
y_test = test_df["label"]
predictions = model.predict(X_test)

# 1. Precision / recall / F1 for each class
print(classification_report(y_test, predictions, target_names=["negative", "positive"]))

# 2. Confusion matrix chart
cm = confusion_matrix(y_test, predictions)
ConfusionMatrixDisplay(cm, display_labels=["negative", "positive"]).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# 3. Words the model relies on most
words = vectorizer.get_feature_names_out()
weights = model.coef_[0]
order = np.argsort(weights)

print("\nMost NEGATIVE words:", [words[i] for i in order[:10]])
print("Most POSITIVE words:", [words[i] for i in order[-10:]])
