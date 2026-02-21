# Startup Success Prediction using Machine Learning

This project predicts whether a startup will be successful or fail based on funding history, milestones, location, category, and relationship-based features.

## Problem Statement
Startup failure is common. This project uses historical startup data to build a machine learning model that predicts startup success using structured features.

## Dataset
- Startup funding, milestones, investor information
- Manually engineered numeric features
- Target variable: `successful` (1 = success, 0 = failure)

## Feature Engineering
- Funding structure indicators (VC, Angel, Round A–D)
- Relationship tiering
- Location and category flags
- Startup age and milestone timing

All categorical features were manually encoded, resulting in a fully numeric dataset.

## Model
- Gradient Boosting Classifier
- Evaluated using accuracy, precision, recall, F1-score, and ROC-AUC
- Final model selected based on balanced performance and reduced false positives

## User Testing
The trained model is deployed using Streamlit for real-time prediction via a user input form.

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
