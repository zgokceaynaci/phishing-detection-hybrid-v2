import re
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)

# === Load Models & Vectorizers === #
with open("models/turkish/naive_bayes.pkl", "rb") as f:
    turkish_model = pickle.load(f)
with open("models/turkish/tfidf_vectorizer.pkl", "rb") as f:
    turkish_vectorizer = pickle.load(f)

with open("models/combined/lightgbm_model.pkl", "rb") as f:
    combined_ml_model = pickle.load(f)
with open("models/combined/tfidf_vectorizer.pkl", "rb") as f:
    combined_vectorizer = pickle.load(f)
combined_dl_model = load_model("models/combined/spam_classifier.keras")
with open("models/combined/tokenizer.pickle", "rb") as f:
    combined_tokenizer = pickle.load(f)

with open("models/url/lightgbm.pkl", "rb") as f:
    url_ml_model = pickle.load(f)
with open("models/url/tfidf_vectorizer.pkl", "rb") as f:
    url_vectorizer = pickle.load(f)
url_dl_model = load_model("models/url/spam_classifier_dl.keras")
with open("models/url/tokenizer_dl.pickle", "rb") as f:
    url_tokenizer = pickle.load(f)

with open("models/ceas/lightgbm.pkl", "rb") as f:
    ceas_ml_model = pickle.load(f)
with open("models/ceas/tfidf_vectorizer.pkl", "rb") as f:
    ceas_vectorizer = pickle.load(f)
ceas_dl_model = load_model("models/ceas/spam_classifier_dl.keras")
with open("models/ceas/tokenizer_dl.pickle", "rb") as f:
    ceas_tokenizer = pickle.load(f)
# === Utilities === #
def is_turkish(text):
    return bool(re.search(r"[çğıöşüÇĞİÖŞÜ]", text))

def is_url(text):
    url_patterns = [
        r'https?://',          # http:// or https://
        r'www\.',              # www.
        r'\.com\b', r'\.net\b', r'\.org\b', r'\.co\b', r'\.xyz\b',
        r'\.info\b', r'\.biz\b', r'\.top\b', r'\.online\b', r'\.ru\b'
    ]
    return any(re.search(p, text.lower()) for p in url_patterns)

def rule_based_check(text):
    spam_keywords_tr = [
        "ödül", "hediye", "kazandınız", "tıklayın", "acele edin", "şimdi satın alın",
        "ücretsiz", "bedava", "hemen kazanın", "para kazanın", "kampanya"
    ]
    spam_keywords_en = [
        "free", "win", "cash", "prize", "offer", "click here", "buy now", "urgent",
        "limited time", "congratulations", "you have won", "act now", "guarantee", "deal"
    ]
    return int(any(kw in text.lower() for kw in (spam_keywords_tr + spam_keywords_en)) or len(text.strip()) < 5)

def predict_dl(text, tokenizer, model):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=150)
    pred = model.predict(padded)[0][0]
    return int(pred > 0.5)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Text is empty"}), 400

    models_output = {
        "rule_based": rule_based_check(text),
        "turkish_nb": None,
        "combined_ml": None,
        "combined_dl": None,
        "url_ml": None,
        "url_dl": None,
        "ceas_ml": None,
        "ceas_dl": None
    }

    # === URL MODE ===
    if is_url(text):
        try:
            vec = url_vectorizer.transform([text])
            models_output["url_ml"] = int(url_ml_model.predict(vec)[0])
        except:
            pass
        try:
            models_output["url_dl"] = predict_dl(text, url_tokenizer, url_dl_model)
        except:
            pass

    # === TURKISH MODE ===
    elif is_turkish(text):
        try:
            vec = turkish_vectorizer.transform([text])
            models_output["turkish_nb"] = int(turkish_model.predict(vec)[0])
        except:
            pass

    # === ENGLISH/GENERIC MODE ===
    else:
        try:
            vec = combined_vectorizer.transform([text])
            models_output["combined_ml"] = int(combined_ml_model.predict(vec)[0])
        except:
            pass
        try:
            models_output["combined_dl"] = predict_dl(text, combined_tokenizer, combined_dl_model)
        except:
            pass
        try:
            vec = ceas_vectorizer.transform([text])
            models_output["ceas_ml"] = int(ceas_ml_model.predict(vec)[0])
        except:
            pass
        try:
            models_output["ceas_dl"] = predict_dl(text, ceas_tokenizer, ceas_dl_model)
        except:
            pass

    # === ENSEMBLE ===
    valid_votes = [v for v in models_output.values() if v is not None]
    vote_sum = sum(valid_votes)
    final_label = "spam" if vote_sum >= len(valid_votes) / 2 else "not_spam"

    return jsonify({
        "final_result": final_label,
        "method": "ensemble-voting",
        "models": models_output
    })


if __name__ == "__main__":
    app.run(debug=True)