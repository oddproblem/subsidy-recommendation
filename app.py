# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from joblib import load
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

app = FastAPI(title="Gov Subsidy ML Recommender API")

# Load preprocessed assets
vectorizer = load("vectorizer.joblib")
X = load("tfidf_matrix.joblib")
with open("cleaned_subsidies.json", "r") as f:
    subsidies = json.load(f)

# Reuse the preprocessing function
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
