from flask import Flask, request, jsonify, render_template
import pickle
import re
from nltk.corpus import stopwords

app = Flask(__name__)

# Load stopwords
stop_words = set(stopwords.words('english'))

# Preprocessing (must match training)
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# 🔥 RULE-BASED SPAM DETECTION (YOUR MAIN GOAL)
def rule_based_check(message):
    message = message.lower()

    spam_patterns = [
        "congratulations",
        "you have won",
        "you are selected",
        "won",
        "winner",
        "prize",
        "claim",
        "free",
        "bonus",
        "urgent",
        "click"
    ]

    score = 0

    for word in spam_patterns:
        if word in message:
            score += 1

    # 🔥 If multiple spam words appear → force spam
    if score >= 2:
        return True

    return False

# Load trained model
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# ✅ FINAL PREDICTION ROUTE
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    message = data["message"]

    if not message.strip():
        return jsonify({"error": "Empty message"}), 400

    # 🔥 RULE-BASED FIRST
    if rule_based_check(message):
        return jsonify({
            "prediction": "Spam",
            "confidence": 99.0,
            "method": "rule-based"
        })

    # 🔥 ML SECOND
    clean_message = preprocess(message)
    X = vectorizer.transform([clean_message])

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    confidence = max(probability)

    result = "Spam" if prediction == 1 else "Ham"

    return jsonify({
        "prediction": result,
        "confidence": round(confidence * 100, 2),
        "method": "machine-learning"
    })

    # 🔥 STEP 2: MACHINE LEARNING
    clean_message = preprocess(message)
    X = vectorizer.transform([clean_message])

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    confidence = max(probability)

    result = "Spam" if prediction == 1 else "Ham"

    return jsonify({
        "prediction": result,
        "confidence": round(confidence * 100, 2),
        "method": "machine-learning"
    })

# Run server
if __name__ == "__main__":
    app.run(debug=True)