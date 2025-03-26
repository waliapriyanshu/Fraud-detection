# Fraud Detection Assignment

This repository contains the implementation of a fraud detection system using machine learning models to classify fraudulent transactions. The project involves data preprocessing, model training, and evaluation of multiple machine learning algorithms.

## Files

- **model.py**: Contains the machine learning models for fraud detection, including Random Forest, Gradient Boosting, XGBoost, and Logistic Regression. It includes model training, hyperparameter tuning, and evaluation.
  
- **preprocess.py**: Handles the data preprocessing steps, including handling missing values, detecting outliers, and addressing multi-collinearity in the dataset.

- **FRAUD DETECTION ASSIGNMENT.pdf**: The assignment document that details the approach, model evaluation, and recommendations for improving the fraud detection system.

## Project Overview

This project aims to identify fraudulent transactions using various machine learning algorithms. The dataset was cleaned and preprocessed before being used for model training. The models used in this assignment are:

1. **Random Forest**
2. **Gradient Boosting**
3. **XGBoost**
4. **Logistic Regression**

Each model was evaluated based on multiple metrics, including accuracy, precision, recall, F1-score, and ROC AUC.

### Preprocessing

1. **Data Cleaning**: 
   - Missing values were handled using imputation techniques.
   - Outliers were detected and removed using Z-scores and box plots.
   - Multi-collinearity was addressed using correlation matrices and Variance Inflation Factor (VIF).
   
2. **Feature Engineering**: 
   - Transaction amount, frequency of transactions, and transaction location were identified as key features for predicting fraud.

### Model Evaluation

All models were trained on a split dataset and evaluated using the following metrics:
- Accuracy: 100% for all models.
- ROC AUC: Close to 1, indicating excellent model performance.
- Precision and Recall: Both models exhibited perfect precision and recall for both classes.

### Key Findings

- **Transaction Amount**: Larger transaction amounts are often linked with fraudulent activities.
- **Transaction Frequency**: Anomalous frequencies of transactions also correlate with fraud.

## Setup and Usage

### Prerequisites
- Python 3.x
- Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`

### Installation

Clone this repository:

```bash
git clone https://github.com/yourusername/fraud-detection.git
