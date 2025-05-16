# src/train/train_email_ml_models.py

import os
import pandas as pd
import pickle
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

# ✅ Paths
DATA_PATH = "data/processed/email_combined_cleaned.csv"
MODEL_DIR = "models/combined"
os.makedirs(MODEL_DIR, exist_ok=True)

# ✅ Load dataset
df = pd.read_csv(DATA_PATH)
print(f"📄 Loaded dataset. Shape: {df.shape}")

# ✅ Split dataset
X = df["text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ✅ SMOTE for balancing
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_tfidf, y_train)

print("✅ SMOTE applied. Class distribution:")
print(pd.Series(y_train_bal).value_counts())

# ✅ Define models
models = {
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(n_jobs=2, random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_jobs=2,random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "NaiveBayes": MultinomialNB()
}

# ✅ Training function
def train_and_evaluate(model, name):
    print(f"\n🚀 Training {name}...")
    start = time.time()
    model.fit(X_train_bal, y_train_bal)
    elapsed = time.time() - start

    preds = model.predict(X_test_tfidf)
    proba = model.predict_proba(X_test_tfidf)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"📊 Results for {name}:")
    print(f"  Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"  Precision: {precision_score(y_test, preds):.4f}")
    print(f"  Recall:    {recall_score(y_test, preds):.4f}")
    print(f"  F1 Score:  {f1_score(y_test, preds):.4f}")
    if proba is not None:
        print(f"  ROC AUC:   {roc_auc_score(y_test, proba):.4f}")
    print(f"  ⏱ Time: {elapsed:.2f} sec")

    # Save model
    model_path = os.path.join(MODEL_DIR, f"{name.lower()}_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"💾 Saved: {model_path}")


# ✅ Train all models
for name, model in models.items():
    train_and_evaluate(model, name)

# ✅ Save vectorizer
vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
with open(vec_path, "wb") as f:
    pickle.dump(vectorizer, f)
print(f"\n💾 TF-IDF vectorizer saved: {vec_path}")
