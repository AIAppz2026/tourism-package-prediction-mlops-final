---
license: other
tags:
- tabular-classification
- tourism
- mlops
- scikit-learn
- bagging-classifier
pipeline_tag: tabular-classification
library_name: scikit-learn
---

# Tourism Package Prediction Model

This model predicts whether a customer is likely to purchase the Wellness Tourism Package offered by "Visit with Us".

## Project Objective

The objective of this project is to build an end-to-end MLOps pipeline for predicting customer purchase likelihood before marketing outreach. The workflow covers data registration, data preparation, model training, experiment tracking, model registration, deployment, and CI/CD automation.

## Selected Model

- Selected Model: Bagging
- Model Type: Tree-based ensemble classifier
- Training Framework: scikit-learn
- Target Variable: ProdTaken

## Prediction Classes

- 0: Customer is not likely to purchase the package
- 1: Customer is likely to purchase the package

## Best Hyperparameters

{
    "classifier__estimator__max_depth": null,
    "classifier__max_samples": 1.0,
    "classifier__n_estimators": 100
}

## Evaluation Metrics

| Metric | Score |
|---|---:|
| accuracy | 0.9395 |
| precision | 0.9291 |
| recall | 0.7421 |
| f1_score | 0.8252 |
| roc_auc | 0.9826 |

## Business Interpretation

The selected model supports the business objective by helping the marketing team identify customers who are more likely to purchase the Wellness Tourism Package before contacting them.

This allows the company to prioritize higher-potential leads, reduce inefficient manual targeting, and improve the use of marketing and sales resources.

## Numerical Features

- Age
- CityTier
- DurationOfPitch
- NumberOfPersonVisiting
- NumberOfFollowups
- PreferredPropertyStar
- NumberOfTrips
- Passport
- PitchSatisfactionScore
- OwnCar
- NumberOfChildrenVisiting
- MonthlyIncome

## Categorical Features

- TypeofContact
- Occupation
- Gender
- ProductPitched
- MaritalStatus
- Designation

## Repository Files

- best_tourism_model.joblib: Saved trained model pipeline including preprocessing and classifier.
- model_metadata.json: Metadata containing model name, tuned parameters, feature details, target column, and evaluation metrics.
- README.md: Model card describing the model and intended usage.

## Intended Use

This model is intended for academic project demonstration and MLOps deployment practice. It is used in a Streamlit application hosted on Hugging Face Spaces.

## Limitations

The model is trained on the provided tourism customer dataset. Performance may change if customer behavior, marketing strategy, or product offerings change over time. The model should be retrained periodically with updated customer data.