# 🌾 Government Subsidy Recommendation API

A machine learning-powered recommendation system that suggests **government subsidy schemes** based on natural language input from users. Built using FastAPI and scikit-learn with TF-IDF & cosine similarity.

## 🔍 What's This?

Say you're a student, a farmer, or a startup founder looking for government help — this API reads your query and recommends relevant schemes based on actual objectives, application processes, and benefits.

> Think ChatGPT for subsidies. Except it's not chatty. Just helpful.

---

## 🚀 Features

- FastAPI backend
- TF-IDF-based text similarity
- Preprocessed and vectorized corpus for blazing speed
- Clean JSON output with scheme name, objective, contact info
- `/recommend` endpoint for getting suggestions
- `/health` endpoint for checking if the service is alive

---

## 🛠️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
