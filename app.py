# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from joblib import load
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import subprocess

app = FastAPI(title="Gov Subsidy ML Recommender API")

# Load or train model assets
def load_assets():
    global vectorizer, X, subsidies
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    try:
        vectorizer = load(os.path.join(BASE_DIR, "vectorizer.joblib"))
        X = load(os.path.join(BASE_DIR, "tfidf_matrix.joblib"))
        with open(os.path.join(BASE_DIR, "cleaned_subsidies.json"), "r") as f:
            subsidies = json.load(f)
    except Exception as e:
        print(f"⚠️ Asset loading failed: {e}")
        print("🔁 Attempting to run train_model.py to generate assets...")
        subprocess.run(["python", os.path.join(BASE_DIR, "train_model.py")], check=True)
        # Try loading again
        vectorizer = load(os.path.join(BASE_DIR, "vectorizer.joblib"))
        X = load(os.path.join(BASE_DIR, "tfidf_matrix.joblib"))
        with open(os.path.join(BASE_DIR, "cleaned_subsidies.json"), "r") as f:
            subsidies = json.load(f)

load_assets()

# Preprocessing function
def preprocess(text: str):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.strip()

# Request schema
class UserQuery(BaseModel):
    query_text: str
    top_k: int = 3

@app.post("/recommend", response_model=List[dict])
def recommend(user: UserQuery):
    if len(user.query_text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Query too short or invalid.")

    processed_query = preprocess(user.query_text)
    user_vec = vectorizer.transform([processed_query])
    similarities = cosine_similarity(user_vec, X).flatten()
    top_indices = similarities.argsort()[-user.top_k:][::-1]

    recommendations = []
    for idx in top_indices:
        s = subsidies[idx]
        recommendations.append({
            "scheme_name": s.get("scheme_name", "Unknown"),
            "similarity_score": float(similarities[idx]),
            "objective": s.get("objective", ""),
            "approach_authority": s.get("approach_authority", ""),
            "application_channel": s.get("application_channel", "")
        })
    return recommendations

@app.get("/health")
def health():
    return {"status": "ok"}
