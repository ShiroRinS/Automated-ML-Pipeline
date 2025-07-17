"""
Model Training Page
Handles feature selection, model training, and hyperparameter tuning
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime
from .data_preprocessing import DataPreprocessor

class ModelTrainingPage:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.data = None
        self.features = None
        self.target = None
        self.selected_features = None
        self.model = None
        self.scaler = None
        self.training_history = []
        self.best_params = None
        self.feature_importances = None
        self.preprocessing_choices = None
        
    def load_data(self, data_path: str, target_column: str = None) -> Dict[str, Any]:
        """Load data and set target column"""
        self.data = pd.read_csv(data_path)
        
        # If target column not specified, assume it's the last column
        self.target = target_column if target_column else self.data.columns[-1]
        self.features = [col for col in self.data.columns if col != self.target]
        
        return {
            'total_rows': len(self.data),
            'features': self.features,
            'target': self.target,
            'target_distribution': self.data[self.target].value_counts().to_dict()
        }
    
    async def get_feature_recommendations(self) -> Dict[str, Any]:
        """Get feature recommendations from Gemini"""
        try:
            print("Starting feature recommendations...")
            if self.data is None:
                raise ValueError("No data loaded. Please load data first.")
            
            print("Getting Gemini suggestions...")
            suggestions = await self.data_handler.get_feature_suggestions(self.data)
            print("Got Gemini suggestions")
            
            print("Calculating feature importance scores...")
            X = self.data[self.features].copy()
            y = self.data[self.target]
            
            print("Removing problematic features...")
            # Specify optional exclusions for known datasets
            exclude_features = [self.target]  # Always exclude target variable
            features_to_use = [f for f in self.features if f not in exclude_features]
            X = X[features_to_use]
            
            # Log missing values
            missing_data = X.isnull().sum()
            print("\nMissing values in features:")
            for col, count in missing_data.items():
                if count > 0:
                    print(f"{col}: {count} missing values ({(count/len(X))*100:.2f}%)")
            print(f"Features to use: {features_to_use}")
        
            print("Identifying feature types...")
            categorical_features = X.select_dtypes(include=['object']).columns
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
            print(f"Categorical features: {list(categorical_features)}")
            print(f"Numeric features: {list(numeric_features)}")
            
            print("Handling numeric features...")
            for col in numeric_features:
                print(f"Processing numeric feature: {col}")
                # Handle infinite values first
                X[col] = X[col].replace([np.inf, -np.inf], np.nan)
                # Calculate median on non-NaN values
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
            
            print("Handling categorical features...")
            for col in categorical_features:
                print(f"Processing categorical feature: {col}")
                # Replace empty strings with NaN
                X[col] = X[col].replace('', np.nan)
                # Fill missing values with mode
                mode_val = X[col].mode()[0] if not X[col].mode().empty else 'Unknown'
                X[col] = X[col].fillna(mode_val)
                # Convert to category codes
                X[col] = pd.Categorical(X[col]).codes
            
            # Check for any remaining NaN values
            if X.isnull().any().any():
                raise ValueError(f"Data still contains NaN values after preprocessing")
                
            print("Training random forest model...")
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            print("Random forest model trained successfully")
            
            # Get feature importances
            importances = pd.DataFrame({
                'feature': features_to_use,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
        
            return {
                'gemini_suggestions': suggestions['raw_suggestions'],
                'feature_importances': importances.to_dict('records'),
                'timestamp': suggestions['timestamp']
            }
        except Exception as e:
            return {
                'gemini_suggestions': f"Error getting recommendations: {str(e)}",
                'feature_importances': [],
                'timestamp': pd.Timestamp.now()
            }
    
    def get_model(self, model_type: str):
        """Get model instance based on model type"""
        model_type = model_type.lower().replace(' ', '_')
        if model_type == 'random_forest':
            return RandomForestClassifier(random_state=42)
        elif model_type == 'decision_tree':
            return DecisionTreeClassifier(random_state=42)
        elif model_type in ['linear_regression', 'logistic_regression']:
            return LogisticRegression(random_state=42)
        else:
            # Default to Random Forest
            return RandomForestClassifier(random_state=42)

    def train_initial_model(self, selected_features: List[str], test_size: float = 0.2, model_type: str = 'random_forest') -> Dict[str, Any]:
        """Train initial model with selected features"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        self.selected_features = selected_features
        
        # Prepare data
        X = self.data[selected_features]
        y = self.data[self.target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = self.get_model(model_type)
        self.model.fit(X_train_scaled, y_train)
        
        # Get predictions and metrics
        y_pred = self.model.predict(X_test_scaled)
        
        # Get feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances = pd.DataFrame({
                'feature': selected_features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        elif hasattr(self.model, 'coef_'):
            # For linear models that use coefficients
            self.feature_importances = pd.DataFrame({
                'feature': selected_features,
                'importance': np.abs(self.model.coef_[0]) if len(self.model.coef_.shape) > 1 else np.abs(self.model.coef_)
            }).sort_values('importance', ascending=False)
        else:
            self.feature_importances = pd.DataFrame({
                'feature': selected_features,
                'importance': [1/len(selected_features)] * len(selected_features)
            })
        
        results = {
            'train_score': self.model.score(X_train_scaled, y_train),
            'test_score': self.model.score(X_test_scaled, y_test),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_importances': self.feature_importances.to_dict('records')
        }
        
        # Save training history
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'initial_training',
            'features': selected_features,
            'results': results
        })
        
        return results
    
    def tune_model(self, param_grid: Dict[str, List[Any]] = None) -> Dict[str, Any]:
        """Tune model hyperparameters"""
        if self.model is None:
            raise ValueError("No model trained. Please train initial model first.")
            
        X = self.data[self.selected_features]
        y = self.data[self.target]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # If no param_grid provided, use default
        if param_grid is None:
            param_grid = self.data_handler._get_parameter_grid(self.model, X)
        
        # Tune model
        tuning_results = self.data_handler.tune_model_parameters(
            RandomForestClassifier(random_state=42),
            X_scaled,
            y
        )
        
        self.best_params = tuning_results['best_params']
        
        # Train model with best parameters
        self.model = RandomForestClassifier(**self.best_params, random_state=42)
        self.model.fit(X_scaled, y)
        
        # Get feature importances
        self.feature_importances = pd.DataFrame({
            'feature': self.selected_features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results = {
            'best_params': self.best_params,
            'best_score': tuning_results['best_score'],
            'cv_results': tuning_results['cv_results'],
            'feature_importances': self.feature_importances.to_dict('records')
        }
        
        # Save training history
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'hyperparameter_tuning',
            'features': self.selected_features,
            'results': results
        })
        
        return results
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get the history of training operations"""
        return self.training_history
    
    def save_model(self, output_dir: str) -> Dict[str, str]:
        """Save model and related artifacts"""
        if self.model is None:
            raise ValueError("No model trained. Please train model first.")
            
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = output_dir / f"model_{timestamp}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save scaler
        scaler_path = output_dir / f"scaler_{timestamp}.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'features': self.selected_features,
            'target': self.target,
            'best_params': self.best_params,
            'feature_importances': self.feature_importances.to_dict('records'),
            'training_history': self.training_history
        }
        
        metadata_path = output_dir / f"metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
            'metadata_path': str(metadata_path)
        }
