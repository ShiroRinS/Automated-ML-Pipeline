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
        
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data by handling missing values and encoding categorical variables
        
        Args:
            data: Input DataFrame
            
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
        
        # Remove problematic features
        print("\nRemoving problematic features...")
        # Remove high missing value columns (>50% missing)
        for col in data_clean.columns:
            if data_clean[col].isna().sum() / len(data_clean) > 0.5:
                data_clean = data_clean.drop(col, axis=1)
                print(f"Dropped {col} due to high missing values")
        
        # Get feature list for training
        features_to_use = [col for col in data_clean.columns if col not in ['PassengerId', 'Name', 'Ticket', 'Cabin']]
        print(f"Features to use: {features_to_use}")
        data_clean = data_clean[features_to_use]
        
        # Identify numeric and categorical features
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
        
        # Handle categorical features
        print("Handling categorical features...")
        for feature in categorical_features:
            print(f"Processing categorical feature: {feature}")
            mode_val = data_clean[feature].mode()[0]
            data_clean[feature] = data_clean[feature].fillna(mode_val)
            # Encode categorical variables
            if feature == 'Sex':
                data_clean[feature] = data_clean[feature].map({'male': 1, 'female': 0})
            elif feature == 'Embarked':
                data_clean[feature] = data_clean[feature].map({'C': 0, 'Q': 1, 'S': 2})
        
        return data_clean
    
    def get_feature_importance(self, data: pd.DataFrame) -> List[Dict[str, float]]:
        """
        Calculate feature importance using Random Forest
        
        Args:
            data: Preprocessed DataFrame
            
        Returns:
            List of dictionaries containing feature names and importance scores
        """
        print("Training random forest model...")
        
        # Separate features and target
        if 'Survived' in data.columns:
            X = data.drop('Survived', axis=1)
            y = data['Survived']
        else:
            # If no target column, use all features
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
    
    async def get_recommendations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get feature recommendations
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary containing recommendations and feature importance
        """
        try:
            print("Starting feature recommendations...")
            
            # Preprocess data
            data_clean = self.preprocess_data(data)
            
            print("Getting Gemini suggestions...")
            # Get Gemini suggestions
            if ENABLE_GEMINI:
                data_description = f"""
                Dataset Summary:
                - Total Samples: {len(data_clean)}
                - Features: {', '.join(data_clean.columns)}
                - Numeric Features: {', '.join(data_clean.select_dtypes(include=['int64', 'float64']).columns)}
                - Target Variable: Survival prediction (0: Did not survive, 1: Survived)
                
                Key Features:
                - Pclass: Passenger class (1st, 2nd, 3rd)
                - Sex: Gender (encoded as 0: Female, 1: Male)
                - Age: Age in years
                - SibSp: Number of siblings/spouses aboard
                - Parch: Number of parents/children aboard
                - Fare: Passenger fare
                - Embarked: Port of embarkation (C: Cherbourg, Q: Queenstown, S: Southampton)
                """
                
                prompt = f"""
Analyze the dataset below and provide feature recommendations in a structured format.

{data_description}

IMPORTANT: First generate a JSON structure exactly like this example, then I will ask you to convert it to markdown.
Replace the example insights with relevant ones for the Titanic dataset while keeping the exact structure:

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
                response = self._gemini_model.generate_content(prompt, generation_config={
                    "temperature": 0.1,  # Lower temperature for more deterministic output
                    "top_p": 0.95,      # Higher top_p to ensure we get complete sections
                    "top_k": 40,
                    "max_output_tokens": 2048,  # Increased to ensure we get complete output
                    "stop_sequences": ["\n\n\n"]  # Stop at triple newlines to better handle markdown
                })
                print("Got Gemini response")
                suggestions = response.text
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
            
            print("Got Gemini suggestions")
            
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
