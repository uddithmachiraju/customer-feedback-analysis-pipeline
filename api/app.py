from flask import Flask, request, render_template
import joblib
from src.preprocessing import clean_text
from src.config import model_saving_path, vectorizer_saving_path

app = Flask(__name__, template_folder = "../templates")

# Load model and vectorizer
model = joblib.load(model_saving_path)
vectorizer = joblib.load(vectorizer_saving_path) 

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form["text"]

        if not text:
            return render_template("index.html", prediction="Please enter some text!")

        # Preprocess
        cleaned_text = clean_text(text)
        vectorized_text = vectorizer.transform([cleaned_text])

        # Predict
        prediction = model.predict(vectorized_text)[0]
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment = sentiment_map.get(prediction, "Unknown")

        return render_template("index.html", prediction=f"Predicted Sentiment: {sentiment}")

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)

# Text	Expected Sentiment
# "The service was terrible and the product broke within a week."	Negative
# "It was okay, not the best but not the worst either."	Neutral
# "Absolutely love it! Will buy again for sure!"	Positive
# "The packaging was damaged but the item was fine."	Neutral / Slight Positive (depending on model)
# "Worst experience ever, will never order again."	Negative
