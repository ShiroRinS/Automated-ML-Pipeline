#!/usr/bin/env python3
"""
Titanic Dataset Preprocessing Script
Prepares the Titanic dataset for the ML pipeline by handling missing values,
encoding categorical variables, and structuring data for training.
"""
import pandas as pd
import numpy as np

def preprocess_titanic_data(input_file='data/titanic.csv', output_file='data/raw_data.csv'):
    """
    Preprocess Titanic dataset for ML pipeline
    
    Args:
        input_file (str): Path to raw Titanic CSV file
        output_file (str): Path to save processed data
    """
    # Load the data
    df = pd.read_csv(input_file)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Select relevant features for survival prediction
    features_to_use = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    target = 'Survived'
    
    # Create working dataset
    data = df[features_to_use + [target]].copy()
    
    print(f"\nSelected features: {features_to_use}")
    print(f"Target variable: {target}")
    print(f"Data shape after feature selection: {data.shape}")
    
    # Handle missing values
    print(f"\nMissing values before preprocessing:")
    print(data.isnull().sum())
    
    # Fill missing ages with median
    data['Age'].fillna(data['Age'].median(), inplace=True)
    
    # Fill missing embarked with most common value
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    
    # Fill missing fare with median
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    
    # Encode categorical variables
    # Sex: male=1, female=0
    data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})
    
    # Embarked: C=0, Q=1, S=2
    data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    
    print(f"\nMissing values after preprocessing:")
    print(data.isnull().sum())
    
    # Reorder columns so target (Survived) is last
    feature_columns = [col for col in data.columns if col != target]
    final_data = data[feature_columns + [target]]
    
    print(f"\nFinal dataset shape: {final_data.shape}")
    print(f"Final columns: {list(final_data.columns)}")
    print(f"\nFirst few rows:")
    print(final_data.head())
    
    print(f"\nData types:")
    print(final_data.dtypes)
    
    print(f"\nTarget distribution:")
    print(final_data[target].value_counts())
    
    # Save processed data
    final_data.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to: {output_file}")
    
    return final_data

if __name__ == "__main__":
    processed_data = preprocess_titanic_data()
