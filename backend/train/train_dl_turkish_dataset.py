# src/train/train_dl_turkish.py

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional, Add
from tensorflow.keras.callbacks import EarlyStopping

# 📁 Paths
DATA_PATH = "../data/processed/turkish_email_cleaned.csv"
EMBEDDING_PATH = "../data/embeddings/glove.6B.100d.txt"
MODEL_DIR = "../models/turkish"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "spam_classifier.keras")
TOKENIZER_SAVE_PATH = os.path.join(MODEL_DIR, "tokenizer.pickle")

# 1️⃣ Dataset
df = pd.read_csv(DATA_PATH)
df.dropna(inplace=True)
X_raw = df["text"].astype(str).tolist()
y_raw = df["label"].astype(int).tolist()

# 2️⃣ Tokenization
max_vocab_size = 20000
max_seq_length = 150

tokenizer = Tokenizer(num_words=max_vocab_size)
tokenizer.fit_on_texts(X_raw)
sequences = tokenizer.texts_to_sequences(X_raw)
X = pad_sequences(sequences, maxlen=max_seq_length)

# 3️⃣ Label encoding
le = LabelEncoder()
y = le.fit_transform(y_raw)

# 4️⃣ Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
print("🧪 y_test class distribution:", np.unique(y_test, return_counts=True))

# 5️⃣ GloVe
embedding_dim = 100
embedding_index = {}
with open(EMBEDDING_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

# 6️⃣ Embedding matrix
word_index = tokenizer.word_index
num_words = min(max_vocab_size, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embedding_dim))

for word, i in word_index.items():
    if i < max_vocab_size:
        vec = embedding_index.get(word)
        if vec is not None:
            embedding_matrix[i] = vec

# 7️⃣ LSTM Model
input_layer = Input(shape=(max_seq_length,))
embedding_layer = Embedding(
    input_dim=num_words,
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    input_length=max_seq_length,
    trainable=False
)(input_layer)

bi_lstm = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
lstm_output = LSTM(128, return_sequences=False)(bi_lstm)
embedding_dense = Dense(128)(embedding_layer[:, -1, :])
skip_connection = Add()([lstm_output, embedding_dense])

dropout = Dropout(0.5)(skip_connection)
dense = Dense(128, activation='relu')(dropout)
output_layer = Dense(1, activation='sigmoid')(dense)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 8️⃣ Model Training
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=64,
    callbacks=[early_stop]
)

# 9️⃣ Grafics
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title("Loss")
plt.legend()
plt.tight_layout()
plt.show()

# 🔟 Evaluate
preds = (model.predict(X_test) > 0.5).astype(int).flatten()
acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds)
rec = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
roc = roc_auc_score(y_test, preds)

print("\n📊 Evaluation Results:")
print(f"  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print(f"  ROC AUC:   {roc:.4f}")

# 🔢 Confusion Matrix
conf_matrix = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=["Not Spam", "Spam"])
disp.plot(cmap="Blues", values_format="d")

# 💾 Save model
model.save(MODEL_SAVE_PATH)
with open(TOKENIZER_SAVE_PATH, "wb") as f:
    pickle.dump(tokenizer, f)
print(f"\n✅ Model saved to {MODEL_SAVE_PATH}")
print(f"✅ Tokenizer saved to {TOKENIZER_SAVE_PATH}")
