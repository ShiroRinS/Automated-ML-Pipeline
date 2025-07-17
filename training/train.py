#!/usr/bin/env python3
"""
ML Quick Prototype - Training Script
"""
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def log_training_event(message, accuracy=None):
    """Log training events to CSV file"""
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'train_logs.csv')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Create log entry
    log_entry = {
        'timestamp': timestamp,
        'message': message,
        'accuracy': accuracy
    }
    
    # Append to CSV
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([log_entry])
    
    df.to_csv(log_file, index=False)


# Generalized preprocessing functions

def preprocess_data(df, choices):
    '''Apply preprocessing based on provided choices'''
    df = df.copy()  # Make a copy to avoid modifying original
    
    # Get categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # By default, prepare to remove all categorical columns unless specified for encoding
    columns_to_keep = list(df.select_dtypes(exclude=['object']).columns)  # Keep numeric columns
    columns_to_encode = choices.get('encoding', {}).keys()
    
    # Handle missing values
    for col, action in choices.get('missing_actions', {}).items():
        if col in df.columns:  # Check if column exists
            if action == 'drop':
                df = df.dropna(subset=[col])
            elif action == 'mean' and df[col].dtype.kind in 'biufc':  # Check if numeric
                df[col] = df[col].fillna(df[col].mean())
            elif action == 'median' and df[col].dtype.kind in 'biufc':  # Check if numeric
                df[col] = df[col].fillna(df[col].median())
    
    # Handle categorical variables
    encoded_columns = []
    for col, method in choices.get('encoding', {}).items():
        if col in categorical_cols:  # Only process if it's actually categorical
            if method == 'label':
                df[col] = df[col].astype('category').cat.codes
                columns_to_keep.append(col)
            elif method == 'onehot':
                # Create dummy variables
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                # Add dummy columns to dataframe
                df = pd.concat([df, dummies], axis=1)
                # Add new columns to keep list
                columns_to_keep.extend(dummies.columns)
            encoded_columns.append(col)
    
    # Keep only specified columns and encoded ones
    df = df[columns_to_keep]
    
    # Scale numeric features
    scale_method = choices.get('scaling', 'none')
    if scale_method != 'none':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:  # Only scale if numeric columns exist
            if scale_method == 'standard':
                scaler = StandardScaler()
            elif scale_method == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

def analyze_dataset_info(df):
    """Analyze dataset and return information about features and missing values."""
    # Analyze missing values
    missing_stats = df.isnull().sum() / len(df) * 100
    missing_info = {
        col: {
            'missing_percentage': pct,
            'suggestion': 'remove' if pct > 50 else 'impute' if pct > 0 else 'keep'
        }
        for col, pct in missing_stats.items()
    }
    
    # Analyze feature types
    feature_info = {}
    for col in df.columns:
        unique_vals = df[col].nunique()
        if df[col].dtype == 'object':
            feature_info[col] = {
                'type': 'categorical',
                'unique_values': unique_vals,
                'encoding_options': ['label', 'onehot'] if unique_vals == 2 else ['onehot']
            }
        else:
            feature_info[col] = {
                'type': 'numeric',
                'unique_values': unique_vals,
                'scaling_options': ['standard', 'minmax', 'none']
            }
    
    return {
        'features': feature_info,
        'missing_values': missing_info,
        'shape': df.shape,
        'columns': list(df.columns)
    }


def analyze_dataset(df):
    """Analyze dataset and provide preprocessing suggestions."""
    print("\nData Quality Analysis:")
    
    # Missing values analysis
    missing_stats = df.isnull().sum() / len(df) * 100
    print("\nMissing Values (%):\n", missing_stats[missing_stats > 0])
    
    # Feature types analysis
    print("\nFeature Types:")
    feature_types = {}
    for col in df.columns:
        unique_vals = df[col].nunique()
        if df[col].dtype == 'object':
            print(f"{col}: Text/Categorical ({unique_vals} unique values)")
            feature_types[col] = 'categorical'
        else:
            print(f"{col}: Numeric ({unique_vals} unique values)")
            feature_types[col] = 'numeric'
    
    return missing_stats, feature_types


def get_preprocessing_choices(df, missing_stats, feature_types):
    """Get user choices for preprocessing steps."""
    choices = {}
    
    # 1. Feature Selection
    print("\n=== Feature Selection ===")
    choices['selected_features'] = get_feature_selection(df)
    
    # 2. Missing Values Strategy
    print("\n=== Missing Values Strategy ===")
    cols_with_missing = missing_stats[missing_stats > 0].index
    choices['missing_strategy'] = {}
    
    if len(cols_with_missing) > 0:
        print("\nFor each feature with missing values, choose handling strategy:")
        for col in cols_with_missing:
            missing_pct = missing_stats[col]
            print(f"\n{col} ({missing_pct:.1f}% missing)")
            strategy = get_user_input(
                "How to handle missing values?",
                ['mean', 'median', 'drop', 'skip']
            )
            choices['missing_strategy'][col] = strategy
    
    # 3. Categorical Encoding
    print("\n=== Categorical Encoding ===")
    categorical_cols = [col for col, type_ in feature_types.items() 
                       if type_ == 'categorical' and col in choices['selected_features']]
    choices['encoding_strategy'] = {}
    
    if categorical_cols:
        print("\nFor each categorical feature, choose encoding strategy:")
        for col in categorical_cols:
            strategy = get_user_input(
                f"Encoding strategy for {col}?",
                ['label', 'onehot', 'skip']
            )
            choices['encoding_strategy'][col] = strategy
    
    # 4. Feature Scaling
    print("\n=== Feature Scaling ===")
    numeric_cols = [col for col, type_ in feature_types.items() 
                    if type_ == 'numeric' and col in choices['selected_features']]
    if numeric_cols:
        choices['scaling'] = get_user_input(
            "Scale numeric features?",
            ['standard', 'minmax', 'none']
        )
    
    return choices


def setup_directories():
    """Setup necessary directories."""
    base_dir = os.path.dirname(__file__)
    dirs = ['logs', 'data', 'artifacts', 'models']
    
    for d in dirs:
        dir_path = os.path.join(base_dir, d)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory ensured: {dir_path}")


def apply_preprocessing(df, choices):
    """Apply preprocessing based on user choices."""
    # Make a copy to avoid modifying original data
    processed_df = df.copy()
    
    # 1. Select features
    processed_df = processed_df[choices['selected_features']]
    
    # 2. Handle missing values
    for col, strategy in choices['missing_strategy'].items():
        if strategy != 'skip' and col in processed_df.columns:
            if strategy == 'drop':
                processed_df = processed_df.dropna(subset=[col])
            elif strategy == 'mean':
                processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
            elif strategy == 'median':
                processed_df[col] = processed_df[col].fillna(processed_df[col].median())
    
    # 3. Encode categorical features
    for col, strategy in choices['encoding_strategy'].items():
        if strategy != 'skip' and col in processed_df.columns:
            if strategy == 'label':
                processed_df[col] = processed_df[col].astype('category').cat.codes
            elif strategy == 'onehot':
                # Get dummies and drop original column
                dummies = pd.get_dummies(processed_df[col], prefix=col)
                processed_df = pd.concat([processed_df, dummies], axis=1)
                processed_df.drop(col, axis=1, inplace=True)
    
    # 4. Scale numeric features if requested
    if choices.get('scaling') in ['standard', 'minmax']:
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        if choices['scaling'] == 'standard':
            scaler = StandardScaler()
        else:  # minmax
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])
    
    return processed_df


def main(data_path, target_column, preprocessing_choices):
    """Main training function, adaptable for CLI and UI interfaces"""
    try:
        log_training_event("Training started")
        setup_directories()
        
        # Load data
        data = pd.read_csv(data_path)
        log_training_event(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
        
        # Analyze dataset
        dataset_info = analyze_dataset_info(data)
        
        # Apply preprocessing
        processed_data = preprocess_data(data.copy(), preprocessing_choices)
        
        # Split the processed data into features and target
        if target_column in processed_data.columns:
            X = processed_data.drop(target_column, axis=1)
            y = processed_data[target_column]
            
            # Ensure target is properly formatted for classification
            if y.dtype != 'int64':
                y = y.astype('int64')
        else:
            return {"error": "Target column not found"}
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save artifacts
        artifacts_dir = os.path.join(os.path.dirname(__file__), 'artifacts')
        os.makedirs(artifacts_dir, exist_ok=True)
        
        today = datetime.now().strftime('%Y%m%d')
        model_path = os.path.join(artifacts_dir, f'model_{today}.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        log_training_event("Training completed successfully", accuracy=accuracy)
        return {
            "accuracy": accuracy,
            "model_path": model_path,
            "processed_data_shape": processed_data.shape
        }
    except Exception as e:
        log_training_event(f"Training failed: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Example usage with default dataset
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'raw_data.csv')
    preprocessing_choices = {
        'missing_actions': {'Age': 'mean'},
        'encoding': {'Sex': 'label'},
        'scaling': 'standard'
    }
    main(data_path, 'Survived', preprocessing_choices)
