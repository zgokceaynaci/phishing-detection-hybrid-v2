import os
import pandas as pd
import pickle
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE

# ✅ Paths
DATA_PATH = "data/processed/turkish_email_cleaned.csv"
MODEL_DIR = "models/turkish"
os.makedirs(MODEL_DIR, exist_ok=True)

# ✅ Load dataset
df = pd.read_csv(DATA_PATH).dropna()
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

# ✅ TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ✅ SMOTE
if len(y_train.unique()) > 1:
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_tfidf, y_train)
    print("✅ SMOTE applied. Class distribution:\n", pd.Series(y_train_bal).value_counts())
else:
    print("⚠️ Skipping SMOTE. Only one class in y_train.")
    X_train_bal, y_train_bal = X_train_tfidf, y_train

# ✅ Train Naive Bayes
model = MultinomialNB()
print("\n🚀 Training naive_bayes...")
start = time.time()
model.fit(X_train_bal, y_train_bal)
duration = time.time() - start

# ✅ Evaluate
preds = model.predict(X_test_tfidf)
proba = model.predict_proba(X_test_tfidf)[:, 1]

print(f"📊 Naive Bayes Results:")
print(f"  Accuracy:  {accuracy_score(y_test, preds):.4f}")
print(f"  Precision: {precision_score(y_test, preds):.4f}")
print(f"  Recall:    {recall_score(y_test, preds):.4f}")
print(f"  F1 Score:  {f1_score(y_test, preds):.4f}")
print(f"  ROC AUC:   {roc_auc_score(y_test, proba):.4f}")
print(f"  ⏱ Time:    {duration:.2f} sec")

# ✅ Save model and vectorizer
with open(os.path.join(MODEL_DIR, "naive_bayes.pkl"), "wb") as f:
    pickle.dump(model, f)
with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

print(f"\n✅ Model and vectorizer saved in {MODEL_DIR}")
