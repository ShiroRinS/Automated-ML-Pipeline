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
    log_file = 'logs/train_logs.csv'
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

def main():
    """Main training function"""
    try:
        log_training_event("Training started")
        
        # Load data
        data = pd.read_csv('data/raw_data.csv')
        log_training_event(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
        
        # Prepare features and target (assuming last column is target)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        log_training_event(f"Model trained successfully", accuracy=accuracy)
        
        # Save artifacts
        today = datetime.now().strftime('%Y%m%d')
        
        # Save model
        model_path = f'artifacts/model_{today}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save scaler
        scaler_path = f'artifacts/scaler_{today}.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save feature names
        features_path = f'artifacts/features_{today}.json'
        feature_info = {
            'feature_names': list(X.columns),
            'n_features': len(X.columns),
            'training_date': today
        }
        with open(features_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        log_training_event(f"Artifacts saved with date: {today}")
        
        print(f"Training completed successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Model saved to: {model_path}")
        
    except Exception as e:
        log_training_event(f"Training failed: {str(e)}")
        print(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
