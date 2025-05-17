# 🛡️ Phishing Detection with Hybrid Model (v2)

A hybrid phishing detection system combining Rule-Based methods, Classical Machine Learning, and Deep Learning models. This full-stack application is developed using **Flask** (backend) and **React** (frontend), which are capable of detecting phishing attempts via email text or URLs in English and Turkish.  

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
```
phishing-detection-hybrid-v2/
├── backend/                # Flask backend - inference and training scripts
│   └── src/
│       ├── datasets/       # Data cleaning utilities
│       └── inference/      # Main prediction logic
├── data/                   # Raw & processed datasets (excluded via .gitignore)
├── models/                 # Trained model files (excluded via .gitignore)
├── frontend/               # React frontend - UI logic
│   └── src/
│       └── components/     # Text input and animated prediction results
└── README.md
```
---

## 🛠️ Installation

### 🔧 Backend (Flask)

```
    cd backend
    pip install -r requirements.txt
    python src/inference/inference_service.py

```

---

## 🌐 Frontend (React)

```
    cd frontend
    npm install
    npm start

```

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
Merged (CEAS + Phishing_Email)

ML Parts:
📄 Loaded dataset. Shape: (57760, 2)
✅ SMOTE applied. Class distribution:
label
1    23313
0    23313

| Model              | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| ------------------ | -------- | --------- | ------ | -------- | ------- |
| LogisticRegression | 0.9829   | 0.9817    | 0.9844 | 0.9830   | 0.9984  |
| RandomForest       | 0.9788   | 0.9849    | 0.9729 | 0.9789   | 0.9979  |
| ExtraTrees         | 0.9790   | 0.9893    | 0.9688 | 0.9789   | 0.9985  |
| LightGBM           | 0.9819   | 0.9789    | 0.9854 | 0.9821   | 0.9987  |
| NaiveBayes         | 0.9574   | 0.9800    | 0.9346 | 0.9568   | 0.9952  |


Deep learning Parts:

| Model              | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| ------------------ | -------- | --------- | ------ | -------- | ------- |
| LSTM + GloVe       | 0.9944   | 0.9963    | 0.9936 | 0.9950   | 0.9945  |

✉️ email_detection_cleaned:

ML Parts:
📄 Loaded dataset. Shape: (18634, 2)
🔍 Label distribution:
 label
0    11322
1     7312
🔍 Training label distribution:
 label
0    9057
1    5850
✅ SMOTE applied. Class distribution:
label
0    9057
1    9057

| Model              | Accuracy | Precision | Recall | F1 Score | ROC AUC | Time (sec)|
| ------------------ | -------- | --------- | ------ | -------- | ------- | --------- |
| LogisticRegression | 0.9635   | 0.9288    | 0.9822 | 0.9548   | 0.9941  | 0.33      |
| RandomForest       | 0.9616   | 0.9353    | 0.9692 | 0.9520   | 0.9913  | 6.36      |
| ExtraTrees         | 0.9662   | 0.9471    | 0.9679 | 0.9574   | 0.9927  | 7.46      |
| LightGBM           | 0.9614   | 0.9263    | 0.9795 | 0.9521   | 0.9943  | 7.48      |
| NaiveBayes         | 0.9573   | 0.9261    | 0.9685 | 0.9468   | 0.9922  | 0.00      |

Deep learning Parts:

| Model              | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| ------------------ | -------- | --------- | ------ | -------- | ------- |
| LSTM + GloVe       | 0.9047   | 0.8991    | 0.8529 | 0.8754   | 0.8956  |


🔗 URL-Based Models
ML Parts:
📄 Loaded dataset. Shape: (549346, 2)
🔍 Label distribution:
 label
0    392924
1    156422
✅ SMOTE applied. Class distribution:
 label
0    314339
1    314339

| Model              | Accuracy | Precision | Recall | F1 Score | ROC AUC | Time (s) |
| ------------------ | -------- | --------- | ------ | -------- | ------- | -------- |
| LogisticRegression | 0.9386   | 0.8971    | 0.8860 | 0.8915   | 0.9833  | 6.39     |
| RandomForest       | 0.9447   | 0.9124    | 0.8916 | 0.9019   | 0.9863  | 221.56   |
| ExtraTrees         | 0.9469   | 0.9177    | 0.8936 | 0.9055   | 0.9864  | 366.55   |
| LightGBM           | 0.9009   | 0.8347    | 0.8130 | 0.8237   | 0.9589  | 31.24    |
| NaiveBayes         | 0.9247   | 0.8517    | 0.8905 | 0.8706   | 0.9812  | 0.06     |

Deep learning Parts:

| Model              | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| ------------------ | -------- | --------- | ------ | -------- | ------- |
| LSTM + GloVe       | 0.9407   | 0.9374    | 0.8486 | 0.8908   | 0.9130  |


🇹🇷 Turkish Dataset

Due to the relatively small size of the Turkish phishing dataset (7,504 records), using complex machine learning or deep learning architectures was not effective or reliable. Thus, we opted for a Naive Bayes classifier, which provided satisfactory performance for the limited data available.

| Model       | Accuracy | F1-Score |
| ----------- | -------- | -------- |
| Naive Bayes | 90.2%    | 89.0%    |


---
## 🖥️ User Interface Preview

Below are screenshots from the phishing detection tool interface. They showcase spam/ham inputs, model outputs, and visual components.  
These images are added via GitHub issue uploads for clarity.  

---

### 📥 Sample Spam Input

<p align="center">
  <img src="https://github.com/user-attachments/assets/332bdaa7-0a93-4f36-95d4-805c4a2a76e9" width="600" alt="Turkish Email Sample Detection">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/742b4c15-20f8-491b-9add-af1f97297742" width="600" alt="Ensemble Output Results">
</p>


---

### 🟢 Non-Spam (Ham) Input Sample

<p align="center">
  <img src="https://github.com/user-attachments/assets/6b51b5c4-fdfb-4f65-9abe-dde67aa3dce8" width="600" alt="Spam Email Input Screen">
</p>

---

### 🔗 URL-Based Detection Sample

<p align="center">
  <img src="https://github.com/user-attachments/assets/678d20f8-d3db-4b28-881e-77498bfb979c" width="600" alt="Ham Email Input Screen">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/3552f802-b95a-4a17-955e-f281cf1a3fff" width="600" alt="URL Input Detection">
</p>

---

### 🇹🇷 Turkish Text Detection Example

<img width="761" alt="Screenshot 2025-05-16 at 17 28 47" src="https://github.com/user-attachments/assets/9241b598-8325-4fb3-9ab2-a8006d9b1126" />


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
```
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
```
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



