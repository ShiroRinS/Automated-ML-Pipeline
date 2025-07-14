"""
Configuration settings
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
TRAINING_DIR = BASE_DIR / "training"
PREDICTION_DIR = BASE_DIR / "prediction"
ARTIFACTS_DIR = TRAINING_DIR / "artifacts"
LOGS_DIR = TRAINING_DIR / "logs"
OUTPUT_DIR = PREDICTION_DIR / "output"

# API Keys
GEMINI_API_KEY = 'AIzaSyDL-p6OrYr5fUKdGHmPCbdNImN4-v9BBcg'

# Feature recommendation settings
ENABLE_GEMINI = GEMINI_API_KEY is not None

def configure_gemini(api_key: str):
    """Configure Gemini API"""
    global GEMINI_API_KEY, ENABLE_GEMINI
    import google.generativeai as genai
    
    GEMINI_API_KEY = api_key
    ENABLE_GEMINI = True
    genai.configure(api_key=GEMINI_API_KEY)
