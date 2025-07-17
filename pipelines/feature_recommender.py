import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, List, Tuple
import google.generativeai as genai
from config import ENABLE_GEMINI, GEMINI_API_KEY

class FeatureRecommender:
    def __init__(self):
        """Initialize the feature recommender"""
        if ENABLE_GEMINI:
            genai.configure(api_key=GEMINI_API_KEY)
            self._gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        
    def preprocess_data(self, data: pd.DataFrame, categorical_encoding: Dict[str, str] = None) -> pd.DataFrame:
        """
        Preprocess the data by handling missing values and encoding categorical variables
        
        Args:
            data: Input DataFrame
            categorical_encoding: Dictionary mapping column names to their encoding method 
                                ('label', 'onehot', or 'exclude'). Default is 'exclude' for all categorical columns.
            
        Returns:
            Preprocessed DataFrame
        """
        data_clean = data.copy()
        
        # Print missing values info
        missing_info = []
        for col in data.columns:
            missing_count = data[col].isna().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(data)) * 100
                missing_info.append(f"{col}: {missing_count} missing values ({missing_pct:.2f}%)")
        if missing_info:
            print("\nMissing values in features:")
            print("\n".join(missing_info))
        
        # Identify numeric and categorical features first, excluding obvious non-numeric columns
        print("Identifying feature types...")
        exclusions = ['PassengerId', 'Name', 'Ticket', 'Cabin']
        categorical_features = data_clean.select_dtypes(include=['object', 'category']).columns.difference(exclusions)
        numeric_features = data_clean.select_dtypes(include=['int64', 'float64']).columns.difference(exclusions)
        print(f"Categorical features: {list(categorical_features)}")
        print(f"Numeric features: {list(numeric_features)}")
        
        # Remove problematic features
        print("\nAnalyzing data quality...")
        # Get missing value statistics
        missing_stats = {}
        for col in data_clean.columns:
            missing_count = data_clean[col].isna().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(data_clean)) * 100
                missing_stats[col] = missing_pct
                print(f"{col}: {missing_count} missing values ({missing_pct:.2f}%)")
                
                # Drop columns with too many missing values (>50%)
                if missing_pct > 50:
                    data_clean = data_clean.drop(col, axis=1)
                    if col in categorical_features:
                        categorical_features = categorical_features.drop(col)
                    if col in numeric_features:
                        numeric_features = numeric_features.drop(col)
                    print(f"Dropped {col} due to high missing values (>{missing_pct:.1f}%)")
        print("Identifying feature types...")
        categorical_features = data_clean.select_dtypes(include=['object']).columns
        numeric_features = data_clean.select_dtypes(include=['int64', 'float64']).columns
        print(f"Categorical features: {list(categorical_features)}")
        print(f"Numeric features: {list(numeric_features)}")
        
        # Handle numeric features
        print("Handling numeric features...")
        for feature in numeric_features:
            print(f"Processing numeric feature: {feature}")
            median_val = data_clean[feature].median()
            data_clean[feature] = data_clean[feature].fillna(median_val)
        
        # Initialize categorical_encoding if not provided
        if categorical_encoding is None:
            categorical_encoding = {}
            
        # Handle categorical features
        print("Handling categorical features...")
        features_to_drop = []
        
        for feature in categorical_features:
            print(f"Processing categorical feature: {feature}")
            # Fill missing values
            mode_val = data_clean[feature].mode()[0]
            data_clean[feature] = data_clean[feature].fillna(mode_val)
            
            # Get encoding method for this feature (default to 'exclude')
            encoding_method = categorical_encoding.get(feature, 'exclude')
            
            if encoding_method == 'exclude':
                features_to_drop.append(feature)
                print(f"Excluding feature: {feature}")
            elif encoding_method == 'label':
                # Label encoding
                unique_values = data_clean[feature].unique()
                encoding_map = {val: idx for idx, val in enumerate(sorted(unique_values))}
                data_clean[feature] = data_clean[feature].map(encoding_map)
                print(f"Label encoded feature: {feature}")
            elif encoding_method == 'onehot':
                # One-hot encoding
                one_hot = pd.get_dummies(data_clean[feature], prefix=feature)
                data_clean = pd.concat([data_clean, one_hot], axis=1)
                features_to_drop.append(feature)  # Drop original column after one-hot encoding
                print(f"One-hot encoded feature: {feature}")
        
        return data_clean
    
    def get_feature_importance(self, data: pd.DataFrame, target_column: str = None) -> List[Dict[str, float]]:
        """
        Calculate feature importance using Random Forest
        
        Args:
            data: Preprocessed DataFrame
            target_column: Name of target column. If None, uses last column.
            
        Returns:
            List of dictionaries containing feature names and importance scores
        """
        print("Training random forest model...")
        
        # Use specified target column or assume last column if not specified
        target_col = target_column if target_column else (data.columns[-1] if len(data.columns) > 0 else None)
        
        if target_col and target_col in data.columns:
            print(f"Using '{target_col}' as target variable")
            X = data.drop(target_col, axis=1)
            y = data[target_col]
            
            # Convert target to numeric if it's categorical
            if y.dtype == 'object' or y.dtype == 'category':
                print(f"Converting categorical target '{target_col}' to numeric")
                # Use label encoding for categorical target
                y = pd.Categorical(y).codes
        else:
            print("No target variable identified, using all features")
            X = data
            y = np.zeros(len(data))  # Dummy target
        
        # Train a random forest model
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance
        importance_scores = []
        for feature, importance in zip(X.columns, rf.feature_importances_):
            importance_scores.append({
                'feature': feature,
                'importance': float(importance)  # Convert numpy float to Python float
            })
        
        # Sort by importance
        importance_scores = sorted(importance_scores, key=lambda x: x['importance'], reverse=True)
        
        return importance_scores
    
    async def get_recommendations(self, data: pd.DataFrame, categorical_encoding: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Get feature recommendations
        
        Args:
            data: Input DataFrame
            categorical_encoding: Dictionary mapping column names to their encoding method
            
        Returns:
            Dictionary containing recommendations and feature importance
        """
        try:
            print("Starting feature recommendations...")
            
            # Preprocess data
            data_clean = self.preprocess_data(data, categorical_encoding)
            
            print("Getting Gemini suggestions...")
            # Get Gemini suggestions
            if ENABLE_GEMINI:
                # Generate dynamic dataset summary
                numeric_features = data_clean.select_dtypes(include=['int64', 'float64']).columns
                categorical_features = data_clean.select_dtypes(include=['object']).columns
                
                # Create feature descriptions
                feature_descriptions = []
                for col in data_clean.columns:
                    col_type = data_clean[col].dtype
                    missing_pct = (data_clean[col].isna().sum() / len(data_clean)) * 100
                    
                    description = f"- {col}: "
                    if col_type in ['int64', 'float64']:
                        description += f"Numeric feature (range: {data_clean[col].min():.2f} to {data_clean[col].max():.2f})"
                    else:
                        unique_vals = data_clean[col].nunique()
                        description += f"Categorical feature ({unique_vals} unique values)"
                    
                    if missing_pct > 0:
                        description += f", {missing_pct:.1f}% missing"
                    
                    feature_descriptions.append(description)
                
                # Create a more detailed dataset summary
                target_info = f"Target Variable: {data_clean.columns[-1]}" if len(data_clean.columns) > 0 else "No target variable identified"
                
                data_description = f"""
                Dataset Summary:
                - Total Samples: {len(data_clean)}
                - Total Features: {len(data_clean.columns)}
                - Numeric Features ({len(numeric_features)}): {', '.join(numeric_features)}
                - Categorical Features ({len(categorical_features)}): {', '.join(categorical_features)}
                - {target_info}
                
                Data Quality:
                {chr(10).join([f'- {col}: {pct:.1f}% missing' for col, pct in missing_stats.items()])} 
                
                Feature Details:
                {chr(10).join(feature_descriptions)}
                """
                
                prompt = f"""
Analyze the dataset below and provide feature recommendations in a structured format.

{data_description}

Your task is to analyze this dataset and recommend features for a machine learning model. Focus on:
1. Feature importance and relevance to prediction
2. Data quality considerations
3. Feature engineering opportunities

IMPORTANT: First generate a JSON structure exactly like this example, then convert it to markdown.
Replace the example insights with relevant ones for this dataset while keeping the exact structure:

{{
    "recommended_features": [
        {{
            "name": "Sex",
            "importance": "Primary factor in survival predictions",
            "reason": "Clear gender-based survival patterns evident in the data"
        }},
        {{
            "name": "Fare",
            "importance": "Strong predictor",
            "reason": "Higher fares correlate with better survival chances"
        }},
        // Add more features in order of importance
    ],
    "selection_tips": [
        "Combine demographic features (Sex, Age) with socioeconomic indicators",
        "Consider interaction between family size features",
        "Prioritize features with low missing value rates"
    ],
    "important_considerations": [
        "Handle missing Age values with appropriate imputation",
        "Watch for correlations between Pclass and Fare",
        "Ensure proper encoding of categorical variables"
    ]
}}

Now, convert your JSON response into this exact markdown format:

### 🎯 Recommended Features (Most Important First)
- **[Feature Name]**: [Importance and reason combined in one line]

### 💡 Feature Selection Tips
- [Each tip on a new line]

### ⚠️ Important Considerations
- [Each consideration on a new line]

Ensure your response includes all three sections with exact headings and emojis. The content should be relevant to the Titanic dataset.
"""
                
            try:
                print("Making Gemini API call...")
                print("Prompt sent to Gemini:")
                print(prompt)
                response = self._gemini_model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.1,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": 2048,
                        "stop_sequences": ["\n\n\n"]
                    },
                    safety_settings={
                        "HARASSMENT": "block_none",
                        "HATE_SPEECH": "block_none",
                        "DANGEROUS_CONTENT": "block_none",
                        "SEXUAL": "block_none",
                        "MEDICAL": "block_none"
                    }
                )
                print("Got Gemini response")
                suggestions = response.text
                print("Full Gemini response:")
                print(suggestions)
                print(f"Suggestion text length: {len(suggestions)}")
                
                # Validate the response has all required sections
                required_sections = ['### 🎯 Recommended Features', '### 💡 Feature Selection Tips', '### ⚠️ Important Considerations']
                for section in required_sections:
                    if section not in suggestions:
                        raise ValueError(f"Missing required section: {section}")

                print("Validated suggestion text format")
            except Exception as e:
                print(f"Gemini API error: {str(e)}")
                suggestions = (
                    "Error getting Gemini suggestions. Please ensure the API response "
                    "includes ### 🎯 Recommended Features, ### 💡 Feature Selection Tips, "
                    "and ### ⚠️ Important Considerations sections. Error details: "
                    f"{str(e)}"
                )
            else:
                suggestions = "Gemini suggestions not available. Please configure GEMINI_API_KEY."
                print("Gemini API not configured")
            
            print("Finished getting suggestions")
            
            # Calculate feature importance
            print("Calculating feature importance scores...")
            importance_scores = self.get_feature_importance(data_clean)
            
            return {
                "gemini_suggestions": suggestions,
                "feature_importances": importance_scores,
                "timestamp": pd.Timestamp.now()
            }
            
        except Exception as e:
            error_msg = f"Error getting recommendations: {str(e)}"
            print(error_msg)
            return {
                "gemini_suggestions": error_msg,
                "feature_importances": [],
                "timestamp": pd.Timestamp.now()
            }
