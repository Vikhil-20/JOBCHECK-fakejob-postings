import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# --- 1. Data Collection ---
try:
    # Ensure 'fake_job_postings.csv' is in the same directory
    df = pd.read_csv("fake_job_postings.csv")
    print("Data collection complete.")
except FileNotFoundError:
    print("Error: 'fake_job_postings.csv' not found. Please download it and place it in the same directory as this script.")
    exit()

# --- 2. Preprocessing ---
# Handle duplicates & missing values
df = df.drop_duplicates()
# Fill all missing values with an empty string for easier text combination
df = df.fillna('')

# Combine all relevant text columns into a single feature
text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
df['combined_text'] = df[text_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

def clean_text(text):
    """A function to clean the text data."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply cleaning to the combined text column
df['cleaned_text'] = df['combined_text'].apply(clean_text)

print("Preprocessing complete.")

# --- 3. Train-Test Split ---
X = df['cleaned_text']
y = df['fraudulent']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 4. TF-IDF Vectorization ---
tfidf = TfidfVectorizer(stop_words='english', max_features=1500)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# --- 5. Model Training ---
model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
model.fit(X_train_tfidf, y_train)
print("Model training completed.")

# --- 6. Model Evaluation ---
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1 Score: {f1*100:.2f}%")

# --- 7. Save Model & Vectorizer ---
# Create a 'model' directory if it doesn't exist
if not os.path.exists("model"):
    os.makedirs("model")

# Save the model and the vectorizer separately
joblib.dump(model, "model/fake_job_model.pkl")
joblib.dump(tfidf, "model/tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully in the 'model' directory.")