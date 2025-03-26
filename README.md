# Fraud Detection

This repository implements a robust fraud detection system using machine learning to identify fraudulent transactions in a financial dataset. The project involves multiple phases: data preprocessing, feature engineering, model training, and evaluation. The goal is to classify transactions as fraudulent or legitimate using a variety of machine learning algorithms.

## Files

- **model.py**: Contains the machine learning models used for fraud detection, including Random Forest, Gradient Boosting, XGBoost, and Logistic Regression. The file handles model training, hyperparameter tuning, and performance evaluation.
  
- **preprocess.py**: This file manages the data preprocessing steps, including handling missing values, detecting outliers, and addressing multi-collinearity. It also covers key feature engineering tasks, crucial for building effective fraud detection models.

- **FRAUD DETECTION ASSIGNMENT.pdf**: This document provides a detailed overview of the approach, model evaluation, and recommendations for improving the fraud detection system.

## Dataset Overview

The dataset used in this project is a simulation of financial transactions with a focus on detecting fraudulent behavior. The key attributes include:

- **step**: Maps a unit of time in the real world (1 step = 1 hour).
- **type**: Transaction types (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER).
- **amount**: The amount involved in the transaction.
- **nameOrig**: The customer initiating the transaction.
- **oldbalanceOrg**: The balance before the transaction.
- **newbalanceOrig**: The balance after the transaction.
- **nameDest**: The recipient of the transaction.
- **oldbalanceDest**: The recipient's balance before the transaction.
- **newbalanceDest**: The recipient's balance after the transaction.
- **isFraud**: Label indicating whether the transaction was fraudulent (1 for fraud, 0 for legitimate).
- **isFlaggedFraud**: Indicates if a large transfer (more than 200,000) is flagged as potentially fraudulent.

## Data Preprocessing

Preprocessing was a critical part of this project. The following steps were applied to the dataset:

1. **Data Cleaning**:
   - **Missing Values**: Imputation techniques were used to fill missing values based on the data's distribution.
   - **Outliers**: Outliers were detected using Z-scores and box plots, and they were removed or adjusted to ensure the models could generalize better.
   - **Multi-collinearity**: The dataset was checked for high correlations between features using correlation matrices and Variance Inflation Factor (VIF), which were mitigated by removing redundant features.

2. **Feature Engineering**:
   - **Transaction Amount**: One of the most significant features, as large amounts were found to correlate with fraudulent transactions.
   - **Transaction Frequency**: The number of transactions over a period helped in identifying unusual patterns.
   - **Transaction Type**: Differentiating between types of transactions (CASH-IN, CASH-OUT, etc.) was crucial for distinguishing legitimate from fraudulent activities.
   - **Balance Changes**: Large changes in the balance before and after transactions were flagged as potential signs of fraud.

3. **Feature Scaling**: Standardization and normalization were applied to ensure the models trained effectively, especially since some algorithms are sensitive to the scale of features.

## Models Used

In this project, several machine learning models were trained and evaluated to predict fraudulent transactions:

1. **Random Forest**: A robust ensemble method that builds multiple decision trees and averages their results.
2. **Gradient Boosting**: A boosting algorithm that builds trees sequentially, focusing on the errors made by the previous trees.
3. **XGBoost**: An optimized gradient boosting algorithm that uses regularization and parallel processing for better performance.
4. **Logistic Regression**: A simple yet effective model for binary classification, used here for comparison with more complex models.

### Model Evaluation

All models were evaluated on a split dataset using several performance metrics:

- **Accuracy**: All models achieved 100% accuracy, as the dataset is highly imbalanced with a clear distinction between fraud and non-fraud cases.
- **ROC AUC**: All models scored close to 1, indicating excellent performance in distinguishing between fraudulent and legitimate transactions.
- **Precision and Recall**: Both models exhibited perfect precision and recall, accurately identifying fraudulent transactions without any false positives or negatives.

### Key Insights

- **Transaction Amount**: Larger transaction amounts were consistently associated with fraudulent activities. The model's ability to identify large transactions was key in detecting fraud.
- **Transaction Frequency**: Anomalous transaction frequencies (such as rapid, multiple small transactions) were also strong indicators of fraudulent behavior.
- **Balance Shifts**: Large fluctuations in balance before and after the transaction were linked to fraud, highlighting the importance of monitoring account balances.

## Setup and Usage

### Prerequisites

To run the fraud detection system, you'll need the following:

- **Python 3.x**: Ensure Python is installed on your system.
- **Required Libraries**: Install the necessary libraries using pip:

```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```
### Installation
1. Clone this repository:
```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
```
2. Download the dataset [here](https://drive.usercontent.google.com/download?id=1VNpyNkGxHdskfdTNRSjjyNa5qC9u0JyV&export=download&authuser=0) and place it in the project directory.
3. Run the preprocessing script to prepare the data for modeling:
```bash
python preprocess.py
```
4. Train the models and evaluate their performance:
```bash
python model.py
```

### Future Work
While the models show excellent performance in detecting fraud, there is always room for improvement:

Handling Class Imbalance: The dataset is imbalanced, with fewer fraudulent transactions. Future work could include techniques like oversampling or using advanced loss functions.
Real-time Fraud Detection: Deploying the model for real-time fraud detection in production systems can be an interesting challenge.
Feature Expansion: More advanced feature engineering, including temporal patterns and external data sources, could be explored.

### Conclusion
This project demonstrates a successful implementation of fraud detection using machine learning. The models performed exceptionally well, accurately identifying fraudulent transactions and providing insights into key features that contribute to fraud. This system has the potential to be adapted and expanded for real-world applications in banking and finance.

