#!/usr/bin/env python3
"""
Test script for training module
"""
import os
from train import analyze_dataset_info, main

def test_analyze():
    # Path to the sample dataset
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'raw_data.csv')
    
    print(f"Step 1: Analyzing dataset at {data_path}")
    print("=" * 50)
    
    # First, let's analyze the dataset
    import pandas as pd
    data = pd.read_csv(data_path)
    dataset_info = analyze_dataset_info(data)
    
    # Print analysis results
    print("\nDataset Shape:", dataset_info['shape'])
    print("\nFeature Information:")
    for col, info in dataset_info['features'].items():
        print(f"\n{col}:")
        print(f"  Type: {info['type']}")
        print(f"  Unique Values: {info['unique_values']}")
        if info['type'] == 'categorical':
            print(f"  Encoding Options: {info['encoding_options']}")
        else:
            print(f"  Scaling Options: {info['scaling_options']}")
    
    print("\nMissing Values Information:")
    for col, info in dataset_info['missing_values'].items():
        if info['missing_percentage'] > 0:
            print(f"\n{col}:")
            print(f"  Missing: {info['missing_percentage']:.2f}%")
            print(f"  Suggestion: {info['suggestion']}")

def test_training():
    print("\nStep 2: Testing training with preprocessing")
    print("=" * 50)
    
    # Path to the sample dataset
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'raw_data.csv')
    
    # Example preprocessing choices
    preprocessing_choices = {
        'missing_actions': {
            'Age': 'mean',  # Fill missing ages with mean
            'Cabin': 'drop',  # Remove rows with missing cabin
            'Embarked': 'drop'  # Remove rows with missing embarked
        },
        'encoding': {
            'Sex': 'label',  # Convert sex to 0/1
            'Pclass': 'onehot'  # Convert class to one-hot encoding
        },
        'scaling': 'standard'  # Standardize numeric features
    }
    
    print("\nNote: All other categorical columns will be removed by default.")
    print("Categorical columns that will be removed: Name, Ticket, Cabin")
    
    # Run training
    print("\nRunning training with following choices:")
    print("Missing value handling:", preprocessing_choices['missing_actions'])
    print("Encoding:", preprocessing_choices['encoding'])
    print("Scaling:", preprocessing_choices['scaling'])
    
    result = main(data_path, 'Survived', preprocessing_choices)
    
    # Print results
    print("\nTraining Results:")
    if "error" in result:
        print("Error:", result["error"])
    else:
        print("Accuracy:", result["accuracy"])
        print("Model saved at:", result["model_path"])
        print("Processed data shape:", result["processed_data_shape"])

if __name__ == "__main__":
    # Run analysis
    test_analyze()
    
    # Run training test
    test_training()
