import os
import json
from pathlib import Path

import joblib
import pandas as pd

from huggingface_hub import HfApi, create_repo
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

HF_USERNAME = "AIApps2026"
MODEL_REPO_ID = "AIApps2026/tourism-package-model-final"
HF_REPOS_PRIVATE = False

PROJECT_DIR = Path(".")
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "ProdTaken"

def main():
    hf_token = os.getenv("HF_TOKEN")
    api = HfApi(token=hf_token)

    print("Starting model training and registration...")

    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data.drop(columns=[TARGET_COLUMN])
    y_train = train_data[TARGET_COLUMN]

    X_test = test_data.drop(columns=[TARGET_COLUMN])
    y_test = test_data[TARGET_COLUMN]

    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X_train.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    classifier = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=None, random_state=42),
        n_estimators=100,
        max_samples=1.0,
        random_state=42
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_pred_proba)), 4)
    }

    model_path = MODEL_DIR / "best_tourism_model.joblib"
    metadata_path = MODEL_DIR / "model_metadata.json"

    joblib.dump(model_pipeline, model_path)

    model_metadata = {
        "best_model_name": "Bagging",
        "best_parameters": {
            "classifier__estimator__max_depth": None,
            "classifier__max_samples": 1.0,
            "classifier__n_estimators": 100
        },
        "metrics": metrics,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "target_column": TARGET_COLUMN,
        "prediction_classes": {
            "0": "Customer is not likely to purchase the package",
            "1": "Customer is likely to purchase the package"
        }
    }

    with open(metadata_path, "w") as file:
        json.dump(model_metadata, file, indent=4)

    print("Model training completed.")
    print("Metrics:", metrics)

    if hf_token:
        create_repo(
            repo_id=MODEL_REPO_ID,
            repo_type="model",
            private=HF_REPOS_PRIVATE,
            exist_ok=True,
            token=hf_token
        )

        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo="best_tourism_model.joblib",
            repo_id=MODEL_REPO_ID,
            repo_type="model",
            token=hf_token,
            commit_message="Update trained model from GitHub Actions"
        )

        api.upload_file(
            path_or_fileobj=str(metadata_path),
            path_in_repo="model_metadata.json",
            repo_id=MODEL_REPO_ID,
            repo_type="model",
            token=hf_token,
            commit_message="Update model metadata from GitHub Actions"
        )

        print("Model and metadata uploaded to Hugging Face Model Hub successfully.")
    else:
        print("HF_TOKEN not found. Skipping Hugging Face upload.")

if __name__ == "__main__":
    main()