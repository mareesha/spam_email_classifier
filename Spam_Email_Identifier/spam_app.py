import streamlit as st
import pickle
import re
from nltk.corpus import stopwords

# Load model
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

stop_words = set(stopwords.words('english'))

# Preprocessing
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# 🔥 RULE-BASED DETECTION
def rule_based_check(message):
    message = message.lower()

    spam_words = [
        "congratulations",
        "won",
        "winner",
        "selected",
        "prize",
        "claim",
        "free",
        "bonus",
        "urgent",
        "click"
    ]

    score = 0
    for word in spam_words:
        if word in message:
            score += 1

    return score >= 2  # 🔥 trigger spam if 2+ words found

# UI
st.title("📩 Spam Detection System")

message = st.text_area("Enter your message")

if st.button("Check"):

    if message.strip() == "":
        st.warning("Please enter a message")

    # 🔥 RULE-BASED FIRST
    elif rule_based_check(message):
        st.error("🚨 Spam Detected (Rule-Based) | Confidence: 99%")

    else:
        clean_message = preprocess(message)
        X = vectorizer.transform([clean_message])

        prediction = model.predict(X)[0]
        prob = model.predict_proba(X)[0]
        confidence = max(prob)

        if prediction == 1:
            st.error(f"🚨 Spam | Confidence: {confidence*100:.2f}%")
        else:
            st.success(f"✅ Ham | Confidence: {confidence*100:.2f}%")