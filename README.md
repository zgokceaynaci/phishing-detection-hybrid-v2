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

    ```
    cd backend
    pip install -r requirements.txt
    python src/inference/inference_service.py

---

## 🌐 Frontend (React)

    ```
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

| File                                                         | Source                          | Records | Status    |
| ------------------------------------------------------------ | ------------------------------- | ------- | --------- |
| `CEAS_08.csv` → `email_dataset_cleaned.csv`                  | CEAS dataset                    | 39,126  | ✅ Success |
| `Phishing_Email.csv` → `email_detection_cleaned.csv`         | Kaggle                          | 18,634  | ✅ Success |
| `phishing_site_urls.csv` → `url_dataset_cleaned.csv`         | Kaggle                          | 549,346 | ✅ Success |
| `turkish_phishing_dataset.csv` → `turkish_email_cleaned.csv` | Manually labeled                | 7,504   | ✅ Success |
| `email_combined_cleaned.csv`                                 | Merged (CEAS + Phishing\_Email) | 57,760  | ✅ Great   |

---

## 📈 Model Performance
All models were evaluated using 80/20 train-test split and SMOTE for balancing.

✉️ Email-Based (Combined Dataset)

| Model              | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| ------------------ | -------- | --------- | ------ | -------- | ------- |
| LogisticRegression | 0.9945   | 0.9968    | 0.9934 | 0.9951   | 0.9996  |
| RandomForest       | 0.9955   | 0.9961    | 0.9959 | 0.9960   | 0.9996  |
| ExtraTrees         | 0.9960   | 0.9982    | 0.9947 | 0.9964   | 0.9998  |
| LightGBM           | 0.9969   | 0.9968    | 0.9977 | 0.9973   | 0.9999  |
| NaiveBayes         | 0.9808   | 0.9969    | 0.9686 | 0.9826   | 0.9988  |
| LSTM + GloVe       | 96.3%    | 95.9%     |


🔗 URL-Based Models

| Model              | Accuracy | Precision | Recall | F1 Score | ROC AUC | Time (s) |
| ------------------ | -------- | --------- | ------ | -------- | ------- | -------- |
| LogisticRegression | 0.9386   | 0.8971    | 0.8860 | 0.8915   | 0.9833  | 6.39     |
| RandomForest       | 0.9447   | 0.9124    | 0.8916 | 0.9019   | 0.9863  | 221.56   |
| ExtraTrees         | 0.9469   | 0.9177    | 0.8936 | 0.9055   | 0.9864  | 366.55   |
| LightGBM           | 0.9009   | 0.8347    | 0.8130 | 0.8237   | 0.9589  | 31.24    |
| NaiveBayes         | 0.9247   | 0.8517    | 0.8905 | 0.8706   | 0.9812  | 0.06     |
| LSTM + Tokenizer   | 98.1%    | 97.7%     |


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

| Input Type         | Models Used                           |
| ------------------ | ------------------------------------- |
| URL                | Rule-Based + URL ML + URL DL + Voting |
| Turkish Text       | Rule-Based + TR NB + Voting           |
| English/Other Text | Rule-Based + All Models + Voting      |


 ---

## 📁 Excluded from Repo
Due to GitHub file size limits, the following are excluded:

📦 Trained models (.pkl, .keras, .pickle)

📁 Full datasets (raw & processed)

Add this to your .gitignore:

     ```
    /backend/data
    /backend/models
    /frontend/node_modules

 ---

## 🧪 Dataset Preparation
Download and merge datasets manually:
    
    ```
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



