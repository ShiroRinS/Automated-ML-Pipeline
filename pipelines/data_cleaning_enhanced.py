"""
Enhanced Data Cleaning Page
Provides interactive data cleaning tools and options with smart suggestions
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .data_handler import DataHandler

class DataCleaningEnhanced:
    def __init__(self):
        self.data_handler = DataHandler()
        self.data = None
        self.cleaned_data = None
        self.cleaning_history = []
        
    def load_data(self, data_path: str) -> Dict[str, Any]:
        """Load data and perform initial analysis with suggestions"""
        self.data = pd.read_csv(data_path)
        return self.analyze_data_quality()
    
    def analyze_data_quality(self) -> Dict[str, Any]:
        """Analyze data quality and return statistics with smart suggestions"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        analysis = {
            'total_rows': len(self.data),
            'total_columns': len(self.data.columns),
            'columns': [],
            'quality_score': 0
        }
        
        total_missing = 0
        
        # Analyze each column
        for col in self.data.columns:
            col_info = {
                'name': col,
                'type': str(self.data[col].dtype),
                'unique_values': self.data[col].nunique(),
                'missing_count': self.data[col].isnull().sum(),
                'suggestions': []
            }
            
            # Calculate missing percentage
            col_info['missing_pct'] = (col_info['missing_count'] / len(self.data)) * 100
            total_missing += col_info['missing_count']
            
            # Analyze numeric columns
            if self.data[col].dtype in ['int64', 'float64']:
                stats = self.data[col].describe()
                col_info.update({
                    'mean': float(stats['mean']),
                    'std': float(stats['std']),
                    'min': float(stats['min']),
                    'max': float(stats['max']),
                    'q1': float(stats['25%']),
                    'q3': float(stats['75%'])
                })
                
                # Check for outliers
                iqr = col_info['q3'] - col_info['q1']
                outlier_count = len(self.data[
                    (self.data[col] < (col_info['q1'] - 1.5 * iqr)) |
                    (self.data[col] > (col_info['q3'] + 1.5 * iqr))
                ])
                col_info['outlier_count'] = outlier_count
                
                # Generate suggestions for numeric columns
                if col_info['missing_pct'] > 0:
                    if col_info['missing_pct'] > 50:
                        col_info['suggestions'].append({
                            'type': 'warning',
                            'message': f'High missing rate ({col_info["missing_pct"]:.1f}%). Consider removing.',
                            'actions': ['Remove column']
                        })
                    else:
                        col_info['suggestions'].append({
                            'type': 'info',
                            'message': f'Missing values ({col_info["missing_pct"]:.1f}%). Choose handling method:',
                            'actions': [
                                'Fill missing with mean',
                                'Fill missing with median',
                                'Remove rows with missing values'
                            ]
                        })
                
                if outlier_count > 0:
                    col_info['suggestions'].append({
                        'type': 'warning',
                        'message': f'Found {outlier_count} outliers.',
                        'actions': [
                            'Cap outliers',
                            'Remove outliers (IQR method)'
                        ]
                    })
                
            # Analyze categorical columns
            elif self.data[col].dtype == 'object':
                value_counts = self.data[col].value_counts()
                col_info.update({
                    'unique_values': len(value_counts),
                    'top_categories': value_counts.head().to_dict()
                })
                
                # Generate suggestions for categorical columns
                if col_info['missing_pct'] > 0:
                    if col_info['missing_pct'] > 50:
                        col_info['suggestions'].append({
                            'type': 'warning',
                            'message': f'High missing rate ({col_info["missing_pct"]:.1f}%). Consider removing.',
                            'actions': ['Remove column']
                        })
                    else:
                        col_info['suggestions'].append({
                            'type': 'info',
                            'message': f'Missing values ({col_info["missing_pct"]:.1f}%). Choose handling method:',
                            'actions': [
                                'Fill missing with mode',
                                'Fill missing with "Unknown"',
                                'Remove rows with missing values'
                            ]
                        })
                
                if col_info['unique_values'] > 10:
                    col_info['suggestions'].append({
                        'type': 'info',
                        'message': 'High cardinality. Consider encoding:',
                        'actions': [
                            'Encode categorical (Label)',
                            'Group rare categories'
                        ]
                    })
                else:
                    col_info['suggestions'].append({
                        'type': 'info',
                        'message': 'Low cardinality. Consider encoding:',
                        'actions': ['Encode categorical (One-hot)']
                    })
            
            analysis['columns'].append(col_info)
        
        # Calculate overall quality score
        total_possible_values = len(self.data) * len(self.data.columns)
        missing_rate = total_missing / total_possible_values
        analysis['quality_score'] = round((1 - missing_rate) * 100, 2)
        
        return analysis
    
    def apply_cleaning(self, column: str, method: str, params: Dict[str, Any] = None) -> Tuple[pd.DataFrame, str]:
        """Apply selected cleaning method to the data"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        if self.cleaned_data is None:
            self.cleaned_data = self.data.copy()
            
        result_message = ""
        data_before = self.cleaned_data.copy()
        
        try:
            # Execute the cleaning operation
            if method == "Remove column":
                self.cleaned_data = self.cleaned_data.drop(column, axis=1)
                result_message = f"Removed column: {column}"
                
            elif method == "Fill missing with mean":
                mean_val = self.cleaned_data[column].mean()
                self.cleaned_data[column].fillna(mean_val, inplace=True)
                result_message = f"Filled missing values in {column} with mean: {mean_val:.2f}"
                
            elif method == "Fill missing with median":
                median_val = self.cleaned_data[column].median()
                self.cleaned_data[column].fillna(median_val, inplace=True)
                result_message = f"Filled missing values in {column} with median: {median_val:.2f}"
                
            elif method == "Fill missing with mode":
                mode_val = self.cleaned_data[column].mode()[0]
                self.cleaned_data[column].fillna(mode_val, inplace=True)
                result_message = f"Filled missing values in {column} with mode: {mode_val}"
                
            elif method == 'Fill missing with "Unknown"':
                self.cleaned_data[column].fillna("Unknown", inplace=True)
                result_message = f"Filled missing values in {column} with 'Unknown'"
                
            elif method == "Remove rows with missing values":
                rows_before = len(self.cleaned_data)
                self.cleaned_data = self.cleaned_data.dropna(subset=[column])
                rows_removed = rows_before - len(self.cleaned_data)
                result_message = f"Removed {rows_removed} rows with missing values in {column}"
                
            elif method == "Cap outliers":
                Q1 = self.cleaned_data[column].quantile(0.25)
                Q3 = self.cleaned_data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.cleaned_data[column] = self.cleaned_data[column].clip(lower_bound, upper_bound)
                result_message = f"Capped outliers in {column} to range [{lower_bound:.2f}, {upper_bound:.2f}]"
                
            elif method == "Remove outliers (IQR method)":
                Q1 = self.cleaned_data[column].quantile(0.25)
                Q3 = self.cleaned_data[column].quantile(0.75)
                IQR = Q3 - Q1
                rows_before = len(self.cleaned_data)
                self.cleaned_data = self.cleaned_data[
                    ~((self.cleaned_data[column] < (Q1 - 1.5 * IQR)) | 
                      (self.cleaned_data[column] > (Q3 + 1.5 * IQR)))
                ]
                rows_removed = rows_before - len(self.cleaned_data)
                result_message = f"Removed {rows_removed} outliers from {column}"
                
            elif method == "Encode categorical (Label)":
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                self.cleaned_data[column] = le.fit_transform(self.cleaned_data[column])
                result_message = f"Applied label encoding to {column}"
                
            elif method == "Encode categorical (One-hot)":
                encoded = pd.get_dummies(self.cleaned_data[column], prefix=column)
                self.cleaned_data = pd.concat([self.cleaned_data.drop(column, axis=1), encoded], axis=1)
                result_message = f"Applied one-hot encoding to {column}"
                
            elif method == "Group rare categories":
                threshold = params.get('threshold', 0.05)  # Default 5% threshold
                value_counts = self.cleaned_data[column].value_counts(normalize=True)
                rare_categories = value_counts[value_counts < threshold].index
                self.cleaned_data[column] = self.cleaned_data[column].replace(rare_categories, 'Other')
                result_message = f"Grouped rare categories in {column} (threshold: {threshold})"
            
            # Record the cleaning step
            self.cleaning_history.append({
                'column': column,
                'method': method,
                'params': params,
                'result_message': result_message
            })
            
            return self.cleaned_data, result_message
            
        except Exception as e:
            # Restore data to state before failed operation
            self.cleaned_data = data_before
            raise ValueError(f"Error applying {method} to {column}: {str(e)}")
    
    def get_cleaning_history(self) -> List[Dict[str, Any]]:
        """Get the history of cleaning operations"""
        return self.cleaning_history
    
    def undo_last_cleaning(self) -> Tuple[pd.DataFrame, str]:
        """Undo the last cleaning operation"""
        if not self.cleaning_history:
            raise ValueError("No cleaning operations to undo")
            
        # Remove the last cleaning step
        self.cleaning_history.pop()
        
        # Rerun all cleaning steps from original data
        self.cleaned_data = self.data.copy()
        for step in self.cleaning_history:
            self.apply_cleaning(step['column'], step['method'], step['params'])
            
        return self.cleaned_data, "Undid last cleaning operation"
    
    def save_cleaned_data(self, output_path: str) -> str:
        """Save the cleaned data to a file"""
        if self.cleaned_data is None:
            raise ValueError("No cleaned data available")
            
        output_path = Path(output_path)
        self.cleaned_data.to_csv(output_path, index=False)
        return f"Cleaned data saved to {output_path}"

    def normalize_data(self) -> pd.DataFrame:
        """Normalize numeric columns to range [0,1] using MinMaxScaler"""
        if self.cleaned_data is None:
            self.cleaned_data = self.data.copy()

        # Select numeric columns only
        numeric_cols = self.cleaned_data.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found for normalization")

        # Apply MinMaxScaler
        scaler = MinMaxScaler()
        self.cleaned_data[numeric_cols] = scaler.fit_transform(self.cleaned_data[numeric_cols])

        # Record the cleaning step
        self.cleaning_history.append({
            'column': 'all numeric',
            'method': 'normalize',
            'params': None,
            'result_message': f"Normalized {len(numeric_cols)} numeric columns"
        })

        return self.cleaned_data

    def scale_features(self) -> pd.DataFrame:
        """Scale numeric features using StandardScaler (mean=0, std=1)"""
        if self.cleaned_data is None:
            self.cleaned_data = self.data.copy()

        # Select numeric columns only
        numeric_cols = self.cleaned_data.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found for scaling")

        # Apply StandardScaler
        scaler = StandardScaler()
        self.cleaned_data[numeric_cols] = scaler.fit_transform(self.cleaned_data[numeric_cols])

        # Record the cleaning step
        self.cleaning_history.append({
            'column': 'all numeric',
            'method': 'scale',
            'params': None,
            'result_message': f"Scaled {len(numeric_cols)} numeric columns"
        })

        return self.cleaned_data
