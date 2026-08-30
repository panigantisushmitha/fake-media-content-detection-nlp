import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Load the dataset
data = pd.read_csv("data.csv")

# Get the news text
X = data["text"]

# Get the labels (real/fake)
y = data["label"]

# Convert text into numbers
vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# Create the machine learning model
model = LogisticRegression()

# Train the model
model.fit(X_vectorized, y)

# Save the vectorizer
joblib.dump(vectorizer, "vectorizer.pkl")

# Save the trained model
joblib.dump(model, "model.pkl")

print("Model trained successfully!")