import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from google.generativeai import GenerativeModel

class DataHandler:
    def __init__(self, suggestion_score_threshold: float = 0.7):
        """
        Initialize DataHandler with a suggestion score threshold for data cleanliness
        
        Args:
            suggestion_score_threshold: Threshold for considering data as clean
        """
        self.suggestion_score_threshold = suggestion_score_threshold
        self._gemini_model = GenerativeModel("gemini-2.5-flash")
    
    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data based on suggestion scores
        
        Args:
            df: Input DataFrame with suggestion scores
            
        Returns:
            Processed DataFrame with handled missing values
        """
        # Copy dataframe to avoid modifying original
        processed_df = df.copy()
        
        # Get rows with high suggestion scores (clean data)
        clean_data = processed_df[processed_df['suggestion_score'] >= self.suggestion_score_threshold]
        
        # Get rows needing preprocessing
        dirty_data = processed_df[processed_df['suggestion_score'] < self.suggestion_score_threshold]
        
        if len(dirty_data) > 0:
            # Use clean data statistics for imputation
            for column in processed_df.columns:
                if column != 'suggestion_score':
                    if processed_df[column].dtype in ['int64', 'float64']:
                        # For numerical columns, use median from clean data
                        median_val = clean_data[column].median()
                        processed_df.loc[processed_df['suggestion_score'] < self.suggestion_score_threshold, column].fillna(median_val, inplace=True)
                    else:
                        # For categorical columns, use mode from clean data
                        mode_val = clean_data[column].mode()[0]
                        processed_df.loc[processed_df['suggestion_score'] < self.suggestion_score_threshold, column].fillna(mode_val, inplace=True)
        
        return processed_df
    
    async def get_feature_suggestions(self, data_sample: pd.DataFrame) -> Dict[str, Any]:
        """
        Use Gemini 2.5-flash to suggest potential feature engineering steps
        
        Args:
            data_sample: Sample of the dataset for analysis
            
        Returns:
            Dictionary containing feature engineering suggestions
        """
        # Prepare data description
        data_description = f"""
        Dataset Summary:
        - Columns: {', '.join(data_sample.columns)}
        - Numeric Features: {', '.join(data_sample.select_dtypes(include=['int64', 'float64']).columns)}
        - Categorical Features: {', '.join(data_sample.select_dtypes(include=['object']).columns)}
        - Sample Size: {len(data_sample)}
        """
        
        # Query Gemini for suggestions
        prompt = f"""
        Based on the following dataset description, suggest feature engineering steps:
        {data_description}
        
        Please provide suggestions in the following areas:
        1. Numeric feature transformations
        2. Categorical encoding methods
        3. Feature interactions
        4. Feature scaling recommendations
        5. Dimension reduction if needed
        
        Format the response as a structured dictionary.
        """
        
        response = await self._gemini_model.generate_content(prompt)
        suggestions = response.text
        
        # Parse and structure the suggestions
        # Note: In a real implementation, you'd want to parse the response more robustly
        return {
            "raw_suggestions": suggestions,
            "timestamp": pd.Timestamp.now()
        }

    def tune_model_parameters(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Perform model tuning based on the data characteristics
        
        Args:
            model: ML model to tune
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary containing tuned parameters
        """
        # Example implementation - extend based on your specific model type
        from sklearn.model_selection import GridSearchCV
        
        # Define parameter grid based on data characteristics
        param_grid = self._get_parameter_grid(model, X_train)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            scoring='accuracy'  # Adjust based on your metric
        )
        
        grid_search.fit(X_train, y_train)
        
        return {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_
        }
    
    def _get_parameter_grid(self, model: Any, X_train: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate parameter grid based on model type and data characteristics
        """
        # Example parameter grid - customize based on your model
        n_features = X_train.shape[1]
        
        base_param_grid = {
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Add model-specific parameters
        if hasattr(model, 'n_estimators'):
            base_param_grid['n_estimators'] = [100, 200, 300]
        
        return base_param_grid
