---
title: Tourism Package Predictor
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Tourism Package Predictor

This Streamlit application predicts whether a customer is likely to purchase the Wellness Tourism Package.

The app loads the registered Bagging Classifier model from the Hugging Face Model Hub, accepts customer details through the sidebar, converts the inputs into a dataframe, and returns the predicted purchase likelihood along with the purchase probability.

The application is part of an Advanced Machine Learning and MLOps project covering data registration, data preparation, model training, model registration, deployment, and CI/CD automation.