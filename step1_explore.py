from datasets import load_dataset
import pandas as pd

# Download the IMDB movie reviews dataset (50,000 reviews)
dataset = load_dataset("stanfordnlp/imdb")

# Convert the train and test parts into pandas tables
train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

print("Training reviews:", len(train_df))
print("Testing reviews:", len(test_df))

# Label 0 = negative, 1 = positive
print("\nLabel counts in training data:")
print(train_df["label"].value_counts())

# Look at a couple of real examples
print("\n--- Example review ---")
print(train_df["text"][0][:300])
print("Label:", train_df["label"][0])