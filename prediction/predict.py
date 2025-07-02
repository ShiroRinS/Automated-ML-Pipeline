#!/usr/bin/env python3
"""
ML Quick Prototype - Prediction Script
"""
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import os
import glob

def load_latest_artifacts():
    """Load the most recent model artifacts"""
    # Get the most recent artifacts based on date in filename
    artifacts_dir = '../training/artifacts'
    
    # Find latest model file
    model_files = glob.glob(f'{artifacts_dir}/model_*.pkl')
    if not model_files:
        raise FileNotFoundError("No model files found in artifacts directory")
    
    latest_model_file = max(model_files, key=os.path.getctime)
    
    # Find corresponding scaler and features files
    date_part = latest_model_file.split('_')[-1].replace('.pkl', '')
    scaler_file = f'{artifacts_dir}/scaler_{date_part}.pkl'
    features_file = f'{artifacts_dir}/features_{date_part}.json'
    
    # Load model
    with open(latest_model_file, 'rb') as f:
        model = pickle.load(f)
    
    # Load scaler
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load feature info
    with open(features_file, 'r') as f:
        feature_info = json.load(f)
    
    print(f"Loaded artifacts from: {date_part}")
    print(f"Model: {latest_model_file}")
    print(f"Features: {feature_info['feature_names']}")
    
    return model, scaler, feature_info

def make_predictions(data, model, scaler, feature_info):
    """Make predictions on new data"""
    # Validate features
    expected_features = feature_info['feature_names']
    
    # Ensure data has the right columns
    if not all(col in data.columns for col in expected_features):
        missing_cols = [col for col in expected_features if col not in data.columns]
        raise ValueError(f"Missing columns in input data: {missing_cols}")
    
    # Select and order features correctly
    X = data[expected_features]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    return predictions, probabilities

def main():
    """Main prediction function"""
    try:
        print("Starting prediction process...")
        
        # Load trained artifacts
        model, scaler, feature_info = load_latest_artifacts()
        
        # Load incoming data
        incoming_data = pd.read_csv('data/incoming_data.csv')
        print(f"Loaded incoming data: {incoming_data.shape[0]} rows")
        
        # Make predictions
        predictions, probabilities = make_predictions(
            incoming_data, model, scaler, feature_info
        )
        
        # Prepare output
        output_data = incoming_data.copy()
        output_data['prediction'] = predictions
        
        # Add probability columns
        for i in range(probabilities.shape[1]):
            output_data[f'probability_class_{i}'] = probabilities[:, i]
        
        # Add metadata
        output_data['prediction_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        output_data['model_date'] = feature_info['training_date']
        
        # Save predictions
        today = datetime.now().strftime('%Y%m%d')
        output_file = f'output/prediction_{today}.csv'
        output_data.to_csv(output_file, index=False)
        
        print(f"Predictions completed successfully!")
        print(f"Output saved to: {output_file}")
        print(f"Predicted {len(predictions)} samples")
        print(f"Class distribution: {np.bincount(predictions)}")
        
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
