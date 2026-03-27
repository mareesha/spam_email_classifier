import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset safely
data = pd.read_csv("dataset.csv", encoding='latin-1', on_bad_lines='skip')
data = data.iloc[:, :2]
data.columns = ["label", "message"]

# Preprocessing
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

data["message"] = data["message"].apply(preprocess)

# Convert labels to numbers
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# Split dataset (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    data["message"],
    data["label"],
    test_size=0.3,
    random_state=42,
    stratify=data["label"]   # 🔥 IMPORTANT
)

# 🔥🔥 UPGRADED VECTORIZER (KEY IMPROVEMENT)
vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1,3),   # 🔥 captures phrases like "you have won"
    min_df=1,
    max_df=0.95
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 🔥 Slightly tuned model
model = MultinomialNB(alpha=0.3)
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy:.4f}\n")

print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("📉 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
with open("spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model saved successfully!")