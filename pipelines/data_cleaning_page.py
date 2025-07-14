"""
Data Cleaning Page
Provides interactive data cleaning tools and options
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from .data_handler import DataHandler

class DataCleaningPage:
    def __init__(self):
        self.data_handler = DataHandler()
        self.data = None
        self.cleaned_data = None
        self.cleaning_history = []
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load data and perform initial analysis"""
        self.data = pd.read_csv(data_path)
        return self.analyze_data_quality()
    
    def analyze_data_quality(self) -> Dict[str, Any]:
        """Analyze data quality and return statistics"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        analysis = {
            'total_rows': len(self.data),
            'total_columns': len(self.data.columns),
            'missing_stats': {},
            'data_types': {},
            'unique_counts': {},
            'numeric_stats': {},
            'categorical_stats': {}
        }
        
        for col in self.data.columns:
            # Missing value statistics
            missing_count = self.data[col].isnull().sum()
            analysis['missing_stats'][col] = {
                'count': missing_count,
                'percentage': (missing_count / len(self.data)) * 100
            }
            
            # Data type information
            analysis['data_types'][col] = str(self.data[col].dtype)
            
            # Unique value counts
            analysis['unique_counts'][col] = self.data[col].nunique()
            
            # Numeric column statistics
            if self.data[col].dtype in ['int64', 'float64']:
                analysis['numeric_stats'][col] = {
                    'mean': self.data[col].mean(),
                    'median': self.data[col].median(),
                    'std': self.data[col].std(),
                    'min': self.data[col].min(),
                    'max': self.data[col].max()
                }
            # Categorical column statistics
            elif self.data[col].dtype == 'object':
                value_counts = self.data[col].value_counts()
                analysis['categorical_stats'][col] = {
                    'most_common': value_counts.index[0] if not value_counts.empty else None,
                    'most_common_count': value_counts.iloc[0] if not value_counts.empty else 0,
                }
        
        return analysis
    
    def get_cleaning_options(self) -> Dict[str, List[str]]:
        """Get available cleaning options for each column"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        options = {}
        for col in self.data.columns:
            col_options = ["Remove column"]
            
            if self.data[col].isnull().any():
                if self.data[col].dtype in ['int64', 'float64']:
                    col_options.extend([
                        "Fill missing with mean",
                        "Fill missing with median",
                        "Fill missing with mode",
                        "Fill missing with custom value",
                        "Remove rows with missing values"
                    ])
                else:
                    col_options.extend([
                        "Fill missing with mode",
                        "Fill missing with 'Unknown'",
                        "Fill missing with custom value",
                        "Remove rows with missing values"
                    ])
            
            if self.data[col].dtype in ['int64', 'float64']:
                col_options.extend([
                    "Remove outliers (IQR method)",
                    "Cap outliers",
                    "Scale values (StandardScaler)",
                    "Scale values (MinMaxScaler)"
                ])
            elif self.data[col].dtype == 'object':
                col_options.extend([
                    "Convert to lowercase",
                    "Remove special characters",
                    "Encode categorical (Label)",
                    "Encode categorical (One-hot)"
                ])
                
            options[col] = col_options
        
        return options
    
    def apply_cleaning(self, column: str, method: str, params: Dict[str, Any] = None) -> Tuple[pd.DataFrame, str]:
        """Apply selected cleaning method to the data"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        if self.cleaned_data is None:
            self.cleaned_data = self.data.copy()
            
        result_message = ""
        data_before = self.cleaned_data.copy()
        
        try:
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
                
            elif method == "Fill missing with 'Unknown'":
                self.cleaned_data[column].fillna("Unknown", inplace=True)
                result_message = f"Filled missing values in {column} with 'Unknown'"
                
            elif method == "Fill missing with custom value":
                if params is None or 'value' not in params:
                    raise ValueError("Custom value not provided")
                self.cleaned_data[column].fillna(params['value'], inplace=True)
                result_message = f"Filled missing values in {column} with custom value: {params['value']}"
                
            elif method == "Remove rows with missing values":
                rows_before = len(self.cleaned_data)
                self.cleaned_data = self.cleaned_data.dropna(subset=[column])
                rows_removed = rows_before - len(self.cleaned_data)
                result_message = f"Removed {rows_removed} rows with missing values in {column}"
                
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
                
            elif method == "Cap outliers":
                Q1 = self.cleaned_data[column].quantile(0.25)
                Q3 = self.cleaned_data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.cleaned_data[column] = self.cleaned_data[column].clip(lower_bound, upper_bound)
                result_message = f"Capped outliers in {column} to range [{lower_bound:.2f}, {upper_bound:.2f}]"
                
            elif method == "Convert to lowercase":
                self.cleaned_data[column] = self.cleaned_data[column].str.lower()
                result_message = f"Converted {column} to lowercase"
                
            elif method == "Remove special characters":
                self.cleaned_data[column] = self.cleaned_data[column].str.replace(r'[^a-zA-Z0-9\s]', '')
                result_message = f"Removed special characters from {column}"
                
            elif method == "Encode categorical (Label)":
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                self.cleaned_data[column] = le.fit_transform(self.cleaned_data[column])
                result_message = f"Applied label encoding to {column}"
                
            elif method == "Encode categorical (One-hot)":
                encoded = pd.get_dummies(self.cleaned_data[column], prefix=column)
                self.cleaned_data = pd.concat([self.cleaned_data.drop(column, axis=1), encoded], axis=1)
                result_message = f"Applied one-hot encoding to {column}"
                
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
