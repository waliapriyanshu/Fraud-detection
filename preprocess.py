import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

def preprocess_fraud_data(
    input_path='/Users/priyanshuwalia7/Downloads/Riverdale/detection_model/Fraud.csv', 
    output_path='/Users/priyanshuwalia7/Downloads/Riverdale/detection_model/new_fraud.csv'
):
    """
    Preprocess fraud detection dataset with advanced feature engineering
    
    Parameters:
    -----------
    input_path : str
        Path to the input CSV file
    output_path : str, optional
        Path to save the preprocessed dataset
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed dataset
    """
    # Verify input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load the dataset
    df = pd.read_csv(input_path)
    
    # Extract origin and destination types
    df['originType'] = df['nameOrig'].apply(lambda x: x[0])
    df['destType'] = df['nameDest'].apply(lambda x: x[0])
    
    # Feature Engineering
    
    # 1. Full amount flag
    df['is_full_amount'] = (df['amount'] == df['oldbalanceOrg']).astype(int)
    
    # 2. Destination and Origin Discrepancy
    df['dest_discrepancy'] = df['amount'] - (df['newbalanceDest'] - df['oldbalanceDest'])
    df['origin_discrepancy'] = df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']
    
    # 3. Transaction Pair
    df['transaction_pair'] = df['originType'] + '→' + df['destType']
    
    # Prepare features and target
    # Identify column types
    numeric_features = [
        'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
        'oldbalanceDest', 'newbalanceDest', 
        'is_full_amount', 'dest_discrepancy', 'origin_discrepancy'
    ]
    categorical_features = ['type', 'originType', 'destType', 'transaction_pair']
    
    # Prepare X and y
    X = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud', 'isFraud'], axis=1)
    y = df['isFraud']
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    # Preprocess the data
    X_processed = preprocessor.fit_transform(X)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_processed, y)
    
    # Get feature names after preprocessing
    onehot_encoder = preprocessor.named_transformers_['cat']
    categorical_feature_names = list(onehot_encoder.get_feature_names_out(categorical_features))
    numeric_feature_names = numeric_features
    feature_names = numeric_feature_names + categorical_feature_names
    
    # Reconstruct DataFrame
    preprocessed_df = pd.DataFrame(
        X_resampled, 
        columns=feature_names
    )
    preprocessed_df['isFraud'] = y_resampled
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save preprocessed data
    preprocessed_df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to: {output_path}")
    
    return preprocessed_df

# Example usage
preprocessed_data = preprocess_fraud_data()
print("Preprocessed Data Shape:", preprocessed_data.shape)
print("\nClass Distribution:")
print(preprocessed_data['isFraud'].value_counts(normalize=True))
print("\nFirst few rows:")
print(preprocessed_data.head())

# Verify file exists and check its size
output_file = '/Users/priyanshuwalia7/Downloads/Riverdale/detection_model/new_fraud.csv'
if os.path.exists(output_file):
    file_size = os.path.getsize(output_file)
    print(f"\nOutput file size: {file_size / (1024 * 1024):.2f} MB")
else:
    print("\nOutput file was not created.")