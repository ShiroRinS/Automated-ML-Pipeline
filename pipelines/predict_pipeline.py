#!/usr/bin/env python3
"""
Automated ML Prediction Pipeline
Handles incoming data: preprocesses, scales, and predicts using the latest model artifacts
"""
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import os
from pathlib import Path

class MLPredictionPipeline:
    def __init__(self, data_path="data/incoming_data.csv", output_path="output", artifacts_dir="artifacts"):
        self.data_path = data_path
        self.output_path = Path(output_path)
        self.artifacts_dir = Path(artifacts_dir)

        # Create output directory if it doesn't exist
        self.output_path.mkdir(exist_ok=True)

    def load_latest_artifacts(self):
        """Load the latest trained model, scaler, and feature information"""
        print("=== Loading Latest Model Artifacts ===")

        # Get the most recent artifacts based on date in filename
        model_files = list(self.artifacts_dir.glob("model_*.pkl"))
        if not model_files:
            raise FileNotFoundError("No model files found in artifacts directory")

        latest_model_file = max(model_files, key=os.path.getctime)
        # Extract training_id from filename (e.g., model_20250702_145919.pkl -> 20250702_145919)
        date_part = '_'.join(latest_model_file.stem.split('_')[1:])

        # Load model
        with open(latest_model_file, 'rb') as f:
            model = pickle.load(f)

        # Load corresponding scaler
        scaler_file = self.artifacts_dir / f"scaler_{date_part}.pkl"
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)

        # Load corresponding feature info
        features_file = self.artifacts_dir / f"features_{date_part}.json"
        with open(features_file, 'r') as f:
            feature_info = json.load(f)

        print(f"Model loaded: {latest_model_file}")
        print(f"Feature info loaded: {features_file}")

        return model, scaler, feature_info

    def preprocess_incoming_data(self, feature_names):
        """Load and preprocess incoming data"""
        print("\n=== Preprocessing Incoming Data ===")
        incoming_data = pd.read_csv(self.data_path)
        print(f"Incoming data: {incoming_data.shape[0]} rows")

        # Ensure incoming data contains all required features
        if not all(feature in incoming_data.columns for feature in feature_names):
            missing_features = [feature for feature in feature_names if feature not in incoming_data.columns]
            raise ValueError(f"Incoming data is missing features: {missing_features}")

        X = incoming_data[feature_names].copy()

        # Handle missing values
        print("Handling missing values...")
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                X[col].fillna(X[col].median(), inplace=True)
            else:
                X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown', inplace=True)

        return X, incoming_data

    def predict(self, X, model, scaler):
        """Make predictions on preprocessed data"""
        print("Making predictions...")
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        return predictions, probabilities

    def save_predictions(self, incoming_data, predictions, probabilities):
        """Save predictions to output file"""
        print("\n=== Saving Predictions ===")

        output_data = incoming_data.copy()
        output_data['prediction'] = predictions

        for i in range(probabilities.shape[1]):
            output_data[f'probability_class_{i}'] = probabilities[:, i]

        output_file = self.output_path / f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_data.to_csv(output_file, index=False)
        print(f"Output saved: {output_file}")

        return output_file

    def run_prediction_pipeline(self):
        """Run the automated prediction pipeline"""
        try:
            # Load the latest artifacts
            model, scaler, feature_info = self.load_latest_artifacts()

            # Preprocess the incoming data
            X, incoming_data = self.preprocess_incoming_data(feature_info['selected_features'])

            # Make predictions
            predictions, probabilities = self.predict(X, model, scaler)

            # Save the predictions
            output_file = self.save_predictions(incoming_data, predictions, probabilities)

            print("\n✅ Prediction Pipeline Completed Successfully!")
            return {
                'success': True,
                'output_file': str(output_file)
            }

        except Exception as e:
            print(f"\n❌ Prediction Pipeline Failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

if __name__ == "__main__":
    # Example usage
    prediction_pipeline = MLPredictionPipeline()
    result = prediction_pipeline.run_prediction_pipeline()
