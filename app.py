from flask import Flask, render_template, request
import joblib
import hashlib


app = Flask(__name__)

# Load the trained model
model = joblib.load("fake_news_model.pkl")

# Load the text vectorizer
vectorizer = joblib.load("tfidf.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    confidence = None
    content_hash = None

    if request.method == "POST":

        news_text = request.form["news_text"]

        # Convert news text into numbers
        news_vector = vectorizer.transform([news_text])

        # Predict real or fake
        prediction = model.predict(news_vector)[0]

        # Get confidence
        probabilities = model.predict_proba(news_vector)[0]
        confidence = round(max(probabilities) * 100, 2)

        # Create SHA-256 hash
        content_hash = hashlib.sha256(
            news_text.encode()
        ).hexdigest()

        if prediction == "real":
            result = "Real News"
        else:
            result = "Fake News"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        content_hash=content_hash
    )


if __name__ == "__main__":
    app.run(debug=True)