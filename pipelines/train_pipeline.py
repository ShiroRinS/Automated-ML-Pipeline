#!/usr/bin/env python3
"""
Enhanced ML Training Pipeline
Provides interactive feature selection, training, evaluation, and comprehensive logging
"""
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class MLTrainingPipeline:
    def __init__(self, data_path="data/raw_data.csv", artifacts_dir="artifacts", logs_dir="logs"):
        self.data_path = data_path
        self.artifacts_dir = Path(artifacts_dir)
        self.logs_dir = Path(logs_dir)
        
        # Create directories if they don't exist
        self.artifacts_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        self.data = None
        self.features = None
        self.target = None
        self.selected_features = None
        self.model = None
        self.scaler = None
        self.training_id = None
        
    def load_data(self):
        """Load and analyze raw data"""
        print("=== Loading Data ===")
        self.data = pd.read_csv(self.data_path)
        print(f"Data loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        
        # Assume last column is target
        self.features = list(self.data.columns[:-1])
        self.target = self.data.columns[-1]
        
        print(f"Target variable: {self.target}")
        print(f"Available features: {len(self.features)}")
        return self.data
    
    def analyze_features(self):
        """Analyze and display feature information for selection"""
        print("\n=== Feature Analysis ===")
        feature_info = []
        
        for i, feature in enumerate(self.features):
            feature_data = self.data[feature]
            info = {
                'index': i + 1,
                'name': feature,
                'type': str(feature_data.dtype),
                'missing': feature_data.isnull().sum(),
                'missing_pct': (feature_data.isnull().sum() / len(feature_data)) * 100,
                'unique_values': feature_data.nunique(),
                'sample_values': list(feature_data.dropna().unique()[:5])
            }
            
            if feature_data.dtype in ['int64', 'float64']:
                info.update({
                    'mean': feature_data.mean(),
                    'std': feature_data.std(),
                    'min': feature_data.min(),
                    'max': feature_data.max()
                })
            
            feature_info.append(info)
        
        # Display feature table
        print(f"{'#':<3} {'Feature Name':<15} {'Type':<10} {'Missing':<8} {'Unique':<8} {'Sample Values':<30}")
        print("-" * 80)
        
        for info in feature_info:
            sample_str = str(info['sample_values'])[:28] + ".." if len(str(info['sample_values'])) > 30 else str(info['sample_values'])
            print(f"{info['index']:<3} {info['name']:<15} {info['type']:<10} {info['missing']:<8} {info['unique_values']:<8} {sample_str:<30}")
        
        return feature_info
    
    def select_features(self, feature_indices=None):
        """Interactive feature selection"""
        print("\n=== Feature Selection ===")
        
        if feature_indices is None:
            print("Select features to use for training:")
            print("Options:")
            print("1. Use all features")
            print("2. Select specific features by number (e.g., 1,2,4,5)")
            print("3. Use recommended features (exclude high missing rate)")
            
            choice = input("Enter your choice (1-3): ").strip()
            
            if choice == "1":
                self.selected_features = self.features.copy()
            elif choice == "2":
                indices_input = input("Enter feature numbers separated by commas: ").strip()
                try:
                    indices = [int(x.strip()) - 1 for x in indices_input.split(',')]
                    self.selected_features = [self.features[i] for i in indices if 0 <= i < len(self.features)]
                except:
                    print("Invalid input, using all features")
                    self.selected_features = self.features.copy()
            elif choice == "3":
                # Recommend features with < 10% missing values
                recommended = []
                for feature in self.features:
                    missing_pct = (self.data[feature].isnull().sum() / len(self.data)) * 100
                    if missing_pct < 10:
                        recommended.append(feature)
                self.selected_features = recommended
            else:
                self.selected_features = self.features.copy()
        else:
            # Use provided indices
            self.selected_features = [self.features[i-1] for i in feature_indices if 1 <= i <= len(self.features)]
        
        print(f"Selected features ({len(self.selected_features)}): {self.selected_features}")
        return self.selected_features
    
    def preprocess_data(self):
        """Preprocess selected features"""
        print("\n=== Data Preprocessing ===")
        
        # Extract selected features and target
        X = self.data[self.selected_features].copy()
        y = self.data[self.target].copy()
        
        # Handle missing values
        print("Handling missing values...")
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                X[col].fillna(X[col].median(), inplace=True)
            else:
                X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown', inplace=True)
        
        # Remove rows with missing target
        mask = ~y.isnull()
        X = X[mask]
        y = y[mask]
        
        print(f"Final dataset: {X.shape[0]} rows, {X.shape[1]} features")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """Train and evaluate model"""
        print("\n=== Model Training ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        y_pred = self.model.predict(X_test_scaled)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        results = {
            'train_score': train_score,
            'test_score': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        print(f"Training Score: {train_score:.4f}")
        print(f"Test Score: {test_score:.4f}")
        print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results, X_train.shape[0], X_test.shape[0]
    
    def save_artifacts(self, results, train_size, test_size):
        """Save model artifacts and metadata"""
        print("\n=== Saving Artifacts ===")
        
        # Generate unique training ID
        self.training_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = self.artifacts_dir / f"model_{self.training_id}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save scaler
        scaler_path = self.artifacts_dir / f"scaler_{self.training_id}.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature metadata
        features_metadata = {
            'training_id': self.training_id,
            'selected_features': self.selected_features,
            'n_features': len(self.selected_features),
            'target_variable': self.target,
            'training_date': datetime.now().isoformat(),
            'data_shape': list(self.data.shape),
            'train_size': train_size,
            'test_size': test_size,
            'model_type': 'RandomForestClassifier',
            'scaler_type': 'StandardScaler'
        }
        
        features_path = self.artifacts_dir / f"features_{self.training_id}.json"
        with open(features_path, 'w') as f:
            json.dump(features_metadata, f, indent=2)
        
        print(f"Model saved: {model_path}")
        print(f"Scaler saved: {scaler_path}")
        print(f"Features saved: {features_path}")
        
        return {
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
            'features_path': str(features_path),
            'training_id': self.training_id
        }
    
    def log_training_session(self, results, train_size, test_size, artifact_paths):
        """Log comprehensive training session information"""
        print("\n=== Logging Training Session ===")
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'training_id': self.training_id,
            'data_path': self.data_path,
            'features_used': self.selected_features,
            'n_features': len(self.selected_features),
            'target_variable': self.target,
            'train_size': train_size,
            'test_size': test_size,
            'total_data_size': train_size + test_size,
            'train_score': results['train_score'],
            'test_score': results['test_score'],
            'cv_mean': results['cv_mean'],
            'cv_std': results['cv_std'],
            'model_path': artifact_paths['model_path'],
            'scaler_path': artifact_paths['scaler_path'],
            'features_path': artifact_paths['features_path'],
            'is_active': True,  # Can be used to mark which model is currently in use
            'notes': f"Training completed successfully with {len(self.selected_features)} features"
        }
        
        # Save to detailed log file
        detailed_log_path = self.logs_dir / f"training_session_{self.training_id}.json"
        with open(detailed_log_path, 'w') as f:
            json.dump(log_entry, f, indent=2)
        
        # Append to master training log
        master_log_path = self.logs_dir / "training_history.csv"
        
        # Convert to flat structure for CSV
        csv_entry = {
            'timestamp': log_entry['timestamp'],
            'training_id': log_entry['training_id'],
            'n_features': log_entry['n_features'],
            'features_list': ','.join(log_entry['features_used']),
            'train_size': log_entry['train_size'],
            'test_size': log_entry['test_size'],
            'train_score': log_entry['train_score'],
            'test_score': log_entry['test_score'],
            'cv_mean': log_entry['cv_mean'],
            'cv_std': log_entry['cv_std'],
            'model_path': log_entry['model_path'],
            'scaler_path': log_entry['scaler_path'],
            'is_active': log_entry['is_active']
        }
        
        # Append to CSV
        if master_log_path.exists():
            df = pd.read_csv(master_log_path)
            df = pd.concat([df, pd.DataFrame([csv_entry])], ignore_index=True)
        else:
            df = pd.DataFrame([csv_entry])
        
        df.to_csv(master_log_path, index=False)
        
        print(f"Detailed log saved: {detailed_log_path}")
        print(f"Master log updated: {master_log_path}")
        
        return log_entry
    
    def run_training_pipeline(self, feature_indices=None):
        """Run the complete training pipeline"""
        print("🚀 Starting ML Training Pipeline")
        print("=" * 50)
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Analyze features
            self.analyze_features()
            
            # Step 3: Select features
            self.select_features(feature_indices)
            
            # Step 4: Preprocess data
            X, y = self.preprocess_data()
            
            # Step 5: Train model
            results, train_size, test_size = self.train_model(X, y)
            
            # Step 6: Save artifacts
            artifact_paths = self.save_artifacts(results, train_size, test_size)
            
            # Step 7: Log session
            log_entry = self.log_training_session(results, train_size, test_size, artifact_paths)
            
            print("\n✅ Training Pipeline Completed Successfully!")
            print(f"Training ID: {self.training_id}")
            print(f"Test Accuracy: {results['test_score']:.4f}")
            print(f"Cross-validation Score: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})")
            
            return {
                'success': True,
                'training_id': self.training_id,
                'results': results,
                'artifacts': artifact_paths,
                'log_entry': log_entry
            }
            
        except Exception as e:
            print(f"\n❌ Training Pipeline Failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

def view_training_history():
    """View training history and help select models"""
    logs_dir = Path("logs")
    master_log_path = logs_dir / "training_history.csv"
    
    if not master_log_path.exists():
        print("No training history found.")
        return None
    
    df = pd.read_csv(master_log_path)
    print("\n=== Training History ===")
    print(df[['timestamp', 'training_id', 'n_features', 'test_score', 'cv_mean', 'is_active']].to_string(index=False))
    
    return df

if __name__ == "__main__":
    # Example usage
    pipeline = MLTrainingPipeline()
    
    # Option 1: Interactive mode
    result = pipeline.run_training_pipeline()
    
    # Option 2: Automated mode with specific features
    # result = pipeline.run_training_pipeline(feature_indices=[1, 2, 3, 4, 5])
    
    # View training history
    view_training_history()
