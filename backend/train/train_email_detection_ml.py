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
DATA_PATH = "data/processed/email_detection_cleaned.csv"
MODEL_DIR = "models/email_detection"
os.makedirs(MODEL_DIR, exist_ok=True)

# ✅ Load dataset
df = pd.read_csv(DATA_PATH)
print(f"📄 Loaded dataset. Shape: {df.shape}")
print("🔍 Label distribution:\n", df["label"].value_counts())

X = df["text"]
y = df["label"]

# ✅ Train/Test split
if len(y.unique()) < 2:
    raise ValueError("❌ Dataset must have at least two classes to train a model.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("🔍 Training label distribution:\n", y_train.value_counts())

# ✅ Vectorization
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ✅ SMOTE
if len(y_train.unique()) > 1:
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_tfidf, y_train)
    print("✅ SMOTE applied. Class distribution:")
    print(pd.Series(y_train_bal).value_counts())
else:
    print("⚠️ Skipping SMOTE. Only one class in y_train.")
    X_train_bal, y_train_bal = X_train_tfidf, y_train

# ✅ Models
models = {
    "logistic_regression": LogisticRegression(),
    "random_forest": RandomForestClassifier(n_jobs=1, random_state=42),
    "extra_trees": ExtraTreesClassifier(n_jobs=1, random_state=42),
    "lightgbm": LGBMClassifier(n_jobs=1, random_state=42),
    "naive_bayes": MultinomialNB()
}

def train_and_save_model(name, model):
    print(f"\n🚀 Training {name}...")
    start = time.time()
    model.fit(X_train_bal, y_train_bal)
    duration = time.time() - start

    preds = model.predict(X_test_tfidf)
    proba = model.predict_proba(X_test_tfidf)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"📊 {name} Results:")
    print(f"  Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"  Precision: {precision_score(y_test, preds):.4f}")
    print(f"  Recall:    {recall_score(y_test, preds):.4f}")
    print(f"  F1 Score:  {f1_score(y_test, preds):.4f}")
    if proba is not None:
        print(f"  ROC AUC:   {roc_auc_score(y_test, proba):.4f}")
    print(f"  ⏱ Time:    {duration:.2f} sec")

    with open(os.path.join(MODEL_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(model, f)

for name, model in models.items():
    train_and_save_model(name, model)

# ✅ Save vectorizer
with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

print(f"\n✅ All models and vectorizer saved in {MODEL_DIR}")
