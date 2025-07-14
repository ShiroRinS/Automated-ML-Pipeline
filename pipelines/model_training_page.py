"""
Model Training Page
Handles feature selection, model training, and hyperparameter tuning
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime
from .data_handler import DataHandler

class ModelTrainingPage:
    def __init__(self):
        self.data_handler = DataHandler()
        self.data = None
        self.features = None
        self.target = None
        self.selected_features = None
        self.model = None
        self.scaler = None
        self.training_history = []
        self.best_params = None
        self.feature_importances = None
        
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
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        suggestions = await self.data_handler.get_feature_suggestions(self.data)
        
        # Calculate feature importance scores
        X = self.data[self.features]
        y = self.data[self.target]
        
        # Train a quick random forest to get feature importances
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importances
        importances = pd.DataFrame({
            'feature': self.features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'gemini_suggestions': suggestions['raw_suggestions'],
            'feature_importances': importances.to_dict('records'),
            'timestamp': suggestions['timestamp']
        }
    
    def train_initial_model(self, selected_features: List[str], test_size: float = 0.2) -> Dict[str, Any]:
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
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Get predictions and metrics
        y_pred = self.model.predict(X_test_scaled)
        
        # Get feature importances
        self.feature_importances = pd.DataFrame({
            'feature': selected_features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
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
