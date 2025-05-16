# 🛡️ Phishing Detection with Hybrid Model (v2)

A hybrid phishing detection system combining Rule-Based methods, Classical Machine Learning, and Deep Learning models. This full-stack application is developed using **Flask** (backend) and **React** (frontend), capable of detecting phishing attempts via email text or URLs in English and Turkish.  

> ✅ Supports both textual and URL-based inputs  
> 🔐 Smart detection: Automatically selects relevant models based on input  

---

## 🌟 Overview

Phishing remains one of the most common and dangerous forms of cyberattacks. This project introduces a **hybrid AI architecture** trained on diverse datasets and capable of making predictions through:

- Keyword-based rule filtering  
- ML models with TF-IDF vectorization  
- DL models using LSTM and GloVe embeddings  
- Final ensemble voting across all relevant models

---

## 🗂️ Project Structure
phishing-detection-hybrid-v2/
├── backend/                  # Flask backend - inference, training scripts
│   ├── src/
│   │   ├── datasets/         # Data cleaning utilities
│   │   └── inference/        # Main prediction logic
│   ├── data/                 # Raw & processed datasets (excluded via .gitignore)
│   └── models/               # Trained model files (excluded via .gitignore)
├── frontend/                 # React frontend - UI logic
│   └── src/components/       # Text input and animated prediction results
└── README.md

---

## 🛠️ Installation

### 🔧 Backend (Flask)

    ```bash
    cd backend
    pip install -r requirements.txt
    python src/inference/inference_service.py

---

## 🌐 Frontend (React)

    ```bash
    cd frontend
    npm install
    npm start

---

## 📊 Model Architecture

The detection process consists of three layers:

🧠 Rule-Based Filter: Fast keyword scan for immediate spam triggers

🤖 ML Classifiers: TF-IDF + LightGBM / Naive Bayes

🧠 Deep Learning: LSTM + pre-trained GloVe embeddings

⚖️ Ensemble Voting: Majority vote from all non-null predictions


---

## 📂 Datasets Used
All datasets were cleaned and preprocessed to contain only text and label columns.
| Source                   | Description             | Link                                                                               |
| ------------------------ | ----------------------- | ---------------------------------------------------------------------------------- |
| Phishing Email Dataset   | English spam emails     | [Kaggle](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) |
| Phishing Site URLs       | Spam and non-spam URLs  | [Kaggle](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls)         |
| Phishing Emails Dataset  | Additional mail samples | [Kaggle](https://www.kaggle.com/datasets/subhajournal/phishingemails)              |
| Turkish Phishing Dataset | Manually labeled emails | Local dataset                                                                      |


---

## 📈 Model Performance
All models were evaluated using 80/20 train-test split and SMOTE for balancing.

✉️ Email Models

| Model        | Accuracy | F1-Score |
| ------------ | -------- | -------- |
| Naive Bayes  | 92.1%    | 91.4%    |
| LightGBM     | 94.7%    | 94.2%    |
| LSTM + GloVe | 96.3%    | 95.9%    |

🔗 URL Models
| Model            | Accuracy | F1-Score |
| ---------------- | -------- | -------- |
| LightGBM         | 97.4%    | 96.8%    |
| LSTM + Tokenizer | 98.1%    | 97.7%    |

🇹🇷 Turkish Dataset
| Model       | Accuracy | F1-Score |
| ----------- | -------- | -------- |
| Naive Bayes | 90.2%    | 89.0%    |

---

##  🖥️ User Interface Preview

✍️ Clean input interface

🔄 Animated loading spinner

📋 Results from each model + final ensemble output

 ---

## 🔐 Smart Detection Logic
Based on input content:

🟢 If it’s a URL → Only URL models + Rule-Based + Ensemble

🔵 If it’s Turkish → Only Turkish NB + Rule-Based + Ensemble

⚪ If English text → All models (Email + CEAS + Rule-Based + Ensemble)

 ---

## 📁 Excluded from Repo
Due to GitHub file size limits, the following are excluded:

📦 Trained models (.pkl, .keras, .pickle)

📁 Full datasets (raw & processed)

Add this to your .gitignore:

     ```bash
    /backend/data
    /backend/models
    /frontend/node_modules

 ---

## 🧪 Dataset Preparation
Download and merge datasets manually:
    
```bash

    import pandas as pd

    email_df = pd.read_csv("Phishing_Email.csv")
    url_df = pd.read_csv("phishing_site_urls.csv")
    extra_df = pd.read_csv("phishingemails.csv")

    merged = pd.concat([email_df, url_df, extra_df], ignore_index=True)
    merged = merged[["text", "label"]]  # Ensure uniform schema
    merged.to_csv("final_dataset.csv", index=False)

 ---

## 🎯 How It Works
User inputs a message or URL

Rule-based keywords are checked

If applicable, ML and/or DL models are triggered

Ensemble voting gives the final decision


 ---

## 🚧 Future Work
 AWS Deployment via Lambda, API Gateway

 Log collection with S3

 Training multilingual transformer models (BERT, RoBERTa)

 Admin panel for retraining with user feedback

 ---



