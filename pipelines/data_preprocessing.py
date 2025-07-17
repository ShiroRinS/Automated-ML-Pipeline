"""
Data Preprocessing Module
Handles data preprocessing for model training
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreprocessor:
    def __init__(self):
        self.data = None
        self.original_data = None
        self.preprocessing_history = []
        self.categorical_encoders = {}
        
    def load_data(self, data_path: str) -> Dict[str, Any]:
        """Load and analyze data"""
        self.original_data = pd.read_csv(data_path)
        self.data = self.original_data.copy()
        
        analysis = self.analyze_data()
        return analysis
        
    def analyze_data(self) -> Dict[str, Any]:
        """Analyze dataset and provide information about features"""
        if self.data is None:
            raise ValueError("No data loaded")
            
        # Analyze data types and missing values
        analysis = {
            'shape': self.data.shape,
            'features': {},
            'missing_values': {}
        }
        
        for col in self.data.columns:
            # Get column type and unique values
            is_numeric = pd.api.types.is_numeric_dtype(self.data[col])
            unique_vals = self.data[col].nunique()
            
            # Analyze missing values
            missing_count = self.data[col].isnull().sum()
            missing_pct = (missing_count / len(self.data)) * 100
            
            # Determine appropriate encoding methods
            if is_numeric:
                encoding_options = ['none']
                if unique_vals < 10:  # For low-cardinality numeric features
                    encoding_options.append('onehot')
            else:
                encoding_options = ['label'] if unique_vals == 2 else ['onehot']
                encoding_options.append('remove')
            
            # Feature information
            analysis['features'][col] = {
                'type': 'numeric' if is_numeric else 'categorical',
                'unique_values': int(unique_vals),
                'encoding_options': encoding_options
            }
            
            # Missing value information
            analysis['missing_values'][col] = {
                'count': int(missing_count),
                'percentage': float(missing_pct),
                'suggestion': 'remove' if missing_pct > 50 else 'impute' if missing_pct > 0 else 'keep'
            }
            
        return analysis
    
    def preprocess_data(self, choices: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply preprocessing based on user choices"""
        if self.data is None:
            raise ValueError("No data loaded")
            
        df = self.data.copy()
        preprocessing_summary = {
            'missing_values_handled': [],
            'encodings_applied': [],
            'features_removed': [],
            'scaling_applied': False
        }
        
        # 1. Handle missing values
        for col, action in choices.get('missing_actions', {}).items():
            if col not in df.columns:
                continue
                
            if action == 'drop':
                df = df.dropna(subset=[col])
                preprocessing_summary['missing_values_handled'].append(f"{col}: dropped rows")
            elif action == 'mean' and df[col].dtype.kind in 'biufc':
                df[col] = df[col].fillna(df[col].mean())
                preprocessing_summary['missing_values_handled'].append(f"{col}: filled with mean")
            elif action == 'median' and df[col].dtype.kind in 'biufc':
                df[col] = df[col].fillna(df[col].median())
                preprocessing_summary['missing_values_handled'].append(f"{col}: filled with median")
        
        # 2. Handle categorical variables
        for col, method in choices.get('encoding', {}).items():
            if col not in df.columns:
                continue
                
            if method == 'label':
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                self.categorical_encoders[col] = encoder
                preprocessing_summary['encodings_applied'].append(f"{col}: label encoding")
            elif method == 'onehot':
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df.drop(col, axis=1, inplace=True)
                preprocessing_summary['encodings_applied'].append(f"{col}: one-hot encoding")
            elif method == 'remove':
                df.drop(col, axis=1, inplace=True)
                preprocessing_summary['features_removed'].append(col)
        
        # 3. Scale numeric features if requested
        if choices.get('scaling') in ['standard', 'minmax']:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                if choices['scaling'] == 'standard':
                    scaler = StandardScaler()
                    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                    preprocessing_summary['scaling_applied'] = 'standard'
                else:  # minmax
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                    preprocessing_summary['scaling_applied'] = 'minmax'
        
        # Save preprocessing steps
        self.preprocessing_history.append({
            'choices': choices,
            'summary': preprocessing_summary
        })
        
        return df, preprocessing_summary
    
    def get_preprocessing_history(self) -> List[Dict[str, Any]]:
        """Get history of preprocessing operations"""
        return self.preprocessing_history
