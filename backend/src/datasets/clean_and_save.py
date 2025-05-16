import pandas as pd
import os
import csv

RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"


os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)


def load_and_clean_csv(file_path, required_columns):
    try:
        df = pd.read_csv(file_path, low_memory=False, encoding="utf-8", on_bad_lines="skip")
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"❌ Missing columns in {file_path}: {missing_columns}")
        df = df[required_columns].dropna().reset_index(drop=True)
        print(f"✅ Loaded and cleaned {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None


def process_email_dataset(df):
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df["sender"] = df["sender"].fillna("")
    df["text"] = df["subject"] + " " + df["body"] + " " + df["sender"]

    if df["label"].dtype in ["int64", "float64"]:
        df["label"] = df["label"].astype(int)
    else:
        df["label"] = df["label"].apply(lambda x: 1 if str(x).strip().lower() in ["phishing", "spam"] else 0)

    return df[["text", "label"]]



def process_phishing_email_dataset(df):
    df.rename(columns={"Email Text": "text", "Email Type": "label"}, inplace=True)
    df["text"] = df["text"].fillna("")
    df["label"] = df["label"].apply(lambda x: 1 if str(x).strip().lower() == "phishing email" else 0)
    return df[["text", "label"]]


def process_url_dataset(df):
    df.rename(columns={"URL": "text", "Label": "label"}, inplace=True)
    df["text"] = df["text"].fillna("")
    df["label"] = df["label"].apply(lambda x: 1 if str(x).strip().lower() == "bad" else 0)
    return df[["text", "label"]]


def process_turkish_dataset(df):
    df["Konu"] = df["Konu"].fillna("")
    df["İçerik"] = df["İçerik"].fillna("")
    df["Gönderen"] = df["Gönderen"].fillna("")
    df["text"] = df["Konu"] + " " + df["İçerik"] + " " + df["Gönderen"]

    df["label"] = df["Kategori"].apply(
        lambda x: 1 if str(x).strip().lower() in ["oltalama"] else 0
    )
    return df[["text", "label"]]


def main():
    combined_dataframes = []

    # CEAS_08.csv
    ceas_file = os.path.join(RAW_DATA_PATH, "CEAS_08.csv")
    ceas_df = load_and_clean_csv(ceas_file, ["subject", "body", "sender", "label"])
    if ceas_df is not None:
        ceas_processed = process_email_dataset(ceas_df)
        ceas_processed.to_csv(os.path.join(PROCESSED_DATA_PATH, "email_dataset_cleaned.csv"), index=False, quoting=csv.QUOTE_ALL)

        combined_dataframes.append(ceas_processed)

    # Phishing_Email.csv
    phishing_email_file = os.path.join(RAW_DATA_PATH, "Phishing_Email.csv")
    phishing_email_df = load_and_clean_csv(phishing_email_file, ["Email Text", "Email Type"])
    if phishing_email_df is not None:
        phishing_email_processed = process_phishing_email_dataset(phishing_email_df)
        phishing_email_processed.to_csv(os.path.join(PROCESSED_DATA_PATH, "email_detection_cleaned.csv"), index=False, quoting=csv.QUOTE_ALL)
        combined_dataframes.append(phishing_email_processed)

    # URL dataset
    url_file = os.path.join(RAW_DATA_PATH, "phishing_site_urls.csv")
    url_df = load_and_clean_csv(url_file, ["URL", "Label"])
    if url_df is not None:
        url_processed = process_url_dataset(url_df)
        url_processed.to_csv(os.path.join(PROCESSED_DATA_PATH, "url_dataset_cleaned.csv"), index=False, quoting=csv.QUOTE_ALL)

    # Turkish phishing dataset
    turkish_file = os.path.join(RAW_DATA_PATH, "turkish_phishing_dataset.csv")
    turkish_df = load_and_clean_csv(turkish_file, ["Konu", "İçerik", "Gönderen", "Kategori"])
    if turkish_df is not None:
        turkish_processed = process_turkish_dataset(turkish_df)
        turkish_processed.to_csv(os.path.join(PROCESSED_DATA_PATH, "turkish_email_cleaned.csv"), index=False, quoting=csv.QUOTE_ALL)

    # Combine English email datasets only
    if combined_dataframes:
        email_combined_df = pd.concat(combined_dataframes, ignore_index=True)
        email_combined_df.to_csv(os.path.join(PROCESSED_DATA_PATH, "email_combined_cleaned.csv"), index=False, quoting=csv.QUOTE_ALL)
        print(f"✅ Combined dataset saved. Shape: {email_combined_df.shape}")

    print("✅ All datasets have been processed and saved.")


if __name__ == "__main__":
    main()