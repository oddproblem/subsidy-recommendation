# train_model.py
import json
import re
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump

# Text cleaning
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.strip()
    return text

# Load JSON data
with open('subsidies.json', 'r') as f:
    subsidies = json.load(f)

# Preprocess features
for sub in subsidies:
    sub['processed_text'] = preprocess(sub['text_features'])

corpus = [sub['processed_text'] for sub in subsidies]

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus)

# Save everything
dump(vectorizer, 'vectorizer.joblib')
dump(X, 'tfidf_matrix.joblib')
with open('cleaned_subsidies.json', 'w') as f:
    json.dump(subsidies, f, indent=2)

print("✅ Model and vectorizer saved successfully.")
