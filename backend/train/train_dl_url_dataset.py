
# train/train_dl_url_dataset.py

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
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ✅ Paths
DATASET_NAME = "url_dataset"
DATA_PATH = f"data/processed/{DATASET_NAME}_cleaned.csv"
EMBEDDING_PATH = "data/embeddings/glove.6B.100d.txt"
MODEL_DIR = f"models/url"
os.makedirs(MODEL_DIR, exist_ok=True)

# ✅ Load data (limit to 100,000 samples to speed up dev time, remove if needed)
df = pd.read_csv(DATA_PATH).dropna()
# df = df.sample(100_000, random_state=42)  # ⚠️ Uncomment this line for quick test
X_raw = df["text"].astype(str).tolist()
y_raw = df["label"].astype(int).tolist()

# ✅ Tokenization
max_vocab_size = 20000
max_seq_length = 150
tokenizer = Tokenizer(num_words=max_vocab_size)
tokenizer.fit_on_texts(X_raw)
sequences = tokenizer.texts_to_sequences(X_raw)
X = pad_sequences(sequences, maxlen=max_seq_length)

# ✅ Label encoding
le = LabelEncoder()
y = le.fit_transform(y_raw)

# ✅ Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# ✅ GloVe embeddings
embedding_dim = 100
embedding_index = {}
with open(EMBEDDING_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

# ✅ Embedding matrix
word_index = tokenizer.word_index
num_words = min(max_vocab_size, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i < max_vocab_size:
        vec = embedding_index.get(word)
        if vec is not None:
            embedding_matrix[i] = vec

# ✅ Simplified Model
input_layer = Input(shape=(max_seq_length,))
embedding_layer = Embedding(input_dim=num_words, output_dim=embedding_dim, weights=[embedding_matrix],
                            input_length=max_seq_length, trainable=False)(input_layer)
bi_lstm = Bidirectional(LSTM(64))(embedding_layer)  # Reduced LSTM size and depth
dropout = Dropout(0.5)(bi_lstm)
dense = Dense(64, activation='relu')(dropout)
output_layer = Dense(1, activation='sigmoid')(dense)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

# ✅ Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
checkpoint = ModelCheckpoint(filepath=os.path.join(MODEL_DIR, "best_model.keras"),
                             monitor='val_loss', save_best_only=True, verbose=1)

# ✅ Train
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=20,
                    batch_size=64,
                    callbacks=[early_stop, reduce_lr, checkpoint],
                    verbose=1)

# ✅ Save final model and tokenizer
model.save(os.path.join(MODEL_DIR, "spam_classifier_dl.keras"))
with open(os.path.join(MODEL_DIR, "tokenizer_dl.pickle"), "wb") as f:
    pickle.dump(tokenizer, f)

# ✅ Evaluation
preds = (model.predict(X_test) > 0.5).astype(int).flatten()
print("\n📊 Evaluation:")
print(f"  Accuracy : {accuracy_score(y_test, preds):.4f}")
print(f"  Precision: {precision_score(y_test, preds):.4f}")
print(f"  Recall   : {recall_score(y_test, preds):.4f}")
print(f"  F1 Score : {f1_score(y_test, preds):.4f}")
print(f"  ROC AUC : {roc_auc_score(y_test, preds):.4f}")

# ✅ Confusion matrix
conf_matrix = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=["Not Spam", "Spam"])
disp.plot(cmap="Blues", values_format="d")
plt.tight_layout()
plt.show()
