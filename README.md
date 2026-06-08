# Tourism Package Prediction using Advanced Machine Learning and MLOps

## Project Objective

This project predicts whether a customer is likely to purchase the Wellness Tourism Package offered by "Visit with Us".

The goal is to support efficient customer targeting before marketing outreach, improve campaign planning, and provide an automated MLOps workflow for model updates and deployment.

## Business Problem

The company wants to identify potential buyers more efficiently instead of relying on manual customer targeting. A predictive model can help prioritize high-potential customers, improve marketing strategy, and support customer acquisition.

## Target Variable

- `ProdTaken`
  - 0: Customer did not purchase the package
  - 1: Customer purchased the package

## MLOps Workflow

The project includes the following stages:

1. Data registration on Hugging Face Dataset Hub
2. Data cleaning and preparation
3. Stratified train-test split
4. Model building and experiment tracking
5. Model registration on Hugging Face Model Hub
6. Streamlit deployment on Hugging Face Spaces
7. CI/CD automation using GitHub Actions

## Selected Model

The final selected model is a Bagging Classifier. It was selected because it provided the best overall balance across the model comparison metrics, especially F1-score, precision, and ROC-AUC.

## Repository Structure

```text
tourism-package-prediction-mlops-final/
├── .github/
│   └── workflows/
│       └── pipeline.yml
├── data/
│   ├── tourism.csv
│   ├── train.csv
│   └── test.csv
├── deployment/
│   ├── app.py
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── README.md
│   └── push_to_hf_space.py
├── model_building/
│   ├── data_preparation.py
│   ├── train_and_register_model.py
│   └── validate_deployment.py
├── models/
│   ├── README.md
│   ├── best_tourism_model.joblib
│   └── model_metadata.json
├── screenshots/
│   └── .gitkeep
├── .gitignore
├── README.md
└── requirements.txt
```

## Public Project Assets

- GitHub Repository: https://github.com/AIAppz2026/tourism-package-prediction-mlops-final
- Hugging Face Dataset: https://huggingface.co/datasets/AIApps2026/tourism-package-data-final
- Hugging Face Model: https://huggingface.co/AIApps2026/tourism-package-model-final
- Hugging Face Space: https://huggingface.co/spaces/AIApps2026/tourism-package-predictor-final

## GitHub Actions

The GitHub Actions workflow is defined in `.github/workflows/pipeline.yml`.

The workflow automates data preparation, model training and registration, deployment validation, deployment to Hugging Face Spaces, and committing generated artifacts back to the repository.

The workflow uses a GitHub repository secret named `HF_TOKEN` for Hugging Face authentication.