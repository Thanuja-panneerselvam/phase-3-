# AI-Powered Credit Card Fraud Detection and Prevention

## Project Overview
This project aims to build an AI-based fraud detection system to identify malicious financial transactions in real-time credit/debit card processing systems. It addresses the critical issue of financial fraud using machine learning techniques.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Abstract](#abstract)
- [System Requirements](#system-requirements)
- [Objectives](#objectives)
- [Dataset Description](#dataset-description)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Deployment](#deployment)
- [Future Scope](#future-scope)


## Problem Statement
Detecting fraudulent transactions using AI to reduce financial fraud in real-time credit/debit card processing systems. It is a binary classification problem with high business relevance in banking and fintech sectors.

## Abstract
The project aims to build an AI-based fraud detection system to identify malicious financial transactions. Using a dataset of transaction data, we applied machine learning models like Random Forest and Logistic Regression. Key preprocessing steps included feature scaling and data balancing. Our system demonstrated high accuracy and ROC-AUC scores, making it suitable for deployment in real-world banking environments.

## System Requirements
- **Hardware**: Min 4GB RAM, Intel i5 or higher
- **Software**: Python 3.8+, scikit-learn, pandas, matplotlib
- **IDE**: Jupyter Notebook or Google Colab

## Objectives
- Predict whether a transaction is fraudulent
- Minimize false negatives to prevent real fraud
- Provide real-time alerts for suspicious activity

## Dataset Description
- **Source**: Synthetic dataset generated using NumPy and pandas
- **Type**: Synthetic
- **Size**: 500 rows × 31 columns
- **Features**: V1–V28 (anonymized), Amount, Time, Class
- **Target variable**: Class (0 = Not Fraud, 1 = Fraud)
- **Fraudulent entries**: 10 (2% of the data)

## Data Preprocessing
- Dropped missing values
- Scaled 'Amount' feature using StandardScaler
- Dropped 'Time' as it has limited predictive value
- Output: Cleaned dataset ready for training

## Exploratory Data Analysis (EDA)
- Fraud transactions are <0.2% of total data
- Strong imbalance in class distribution
- Features V14, V10, and V17 showed correlation with fraud
- Plotted heatmap, distribution plots, and boxplots

## Feature Engineering
- Removed less informative features (like 'Time')
- Scaled 'Amount' for better model convergence
- Selected top contributing features based on correlation

## Model Building
- **Models used**: Logistic Regression, Random Forest
- Random Forest chosen for its higher accuracy and robustness
- Trained using train_test_split with 80:20 ratio

## Model Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC AUC
- Random Forest AUC: ~0.98, High recall for fraud class
- Confusion matrix and ROC curves used for analysis

## Deployment
- **Deployment method**: Streamlit Cloud
- **Public link**: [Insert your Streamlit link]
- UI includes file upload and prediction display
- Outputs 'Fraud' or 'Not Fraud' for input transactions

## Future Scope
- Integrate deep learning models like LSTM for sequential transaction analysis
- Use real-time transaction APIs for continuous monitoring
- Improve data handling for unbalanced datasets (e.g., SMOTE)

