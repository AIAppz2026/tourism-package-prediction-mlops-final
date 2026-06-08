import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download

HF_USERNAME = "AIApps2026"
DATASET_REPO_ID = "AIApps2026/tourism-package-data-final"

PROJECT_DIR = Path(".")
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "ProdTaken"

def main():
    hf_token = os.getenv("HF_TOKEN")
    api = HfApi(token=hf_token)

    print("Starting data preparation...")

    try:
        print("Downloading raw dataset from Hugging Face Dataset Hub...")
        raw_dataset_path = hf_hub_download(
            repo_id=DATASET_REPO_ID,
            filename="data/tourism.csv",
            repo_type="dataset",
            token=hf_token
        )
    except Exception as error:
        print("Could not download raw dataset from Hugging Face. Falling back to local data/tourism.csv.")
        print("Download error:", error)
        raw_dataset_path = DATA_DIR / "tourism.csv"

    df = pd.read_csv(raw_dataset_path)
    print("Raw dataset shape:", df.shape)

    df_cleaned = df.copy()

    columns_to_drop = ["Unnamed: 0", "CustomerID"]
    existing_columns_to_drop = [column for column in columns_to_drop if column in df_cleaned.columns]
    df_cleaned = df_cleaned.drop(columns=existing_columns_to_drop)

    df_cleaned["Gender"] = df_cleaned["Gender"].replace({"Fe Male": "Female"})
    df_cleaned["MaritalStatus"] = df_cleaned["MaritalStatus"].replace({"Unmarried": "Single"})

    X = df_cleaned.drop(columns=[TARGET_COLUMN])
    y = df_cleaned[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train

    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test

    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Training dataset saved locally:", train_path, train_df.shape)
    print("Testing dataset saved locally:", test_path, test_df.shape)

    if hf_token:
        api.upload_file(
            path_or_fileobj=str(train_path),
            path_in_repo="data/train.csv",
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            token=hf_token,
            commit_message="Update prepared training dataset from GitHub Actions"
        )

        api.upload_file(
            path_or_fileobj=str(test_path),
            path_in_repo="data/test.csv",
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            token=hf_token,
            commit_message="Update prepared testing dataset from GitHub Actions"
        )

        print("Prepared train and test datasets uploaded to Hugging Face successfully.")
    else:
        print("HF_TOKEN not found. Skipping Hugging Face upload.")

if __name__ == "__main__":
    main()