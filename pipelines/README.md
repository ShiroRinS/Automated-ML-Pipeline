# 🚀 Enhanced ML Pipeline Architecture

This directory contains the enhanced ML pipeline system with two separate, well-designed pipelines for training and prediction.

## 📁 Architecture Overview

```
pipelines/
├── train_pipeline.py      # Interactive training pipeline with feature selection
├── predict_pipeline.py    # Automated prediction pipeline
├── pipeline_manager.py    # Main interface and pipeline coordinator
└── README.md             # This documentation
```

## 🔧 Pipeline 1: Training Pipeline

**File**: `train_pipeline.py`

### Features:
- **Interactive Feature Selection Interface**: Shows all available features with statistics
- **Smart Preprocessing**: Handles missing values and data quality issues
- **Comprehensive Evaluation**: Cross-validation, confusion matrix, classification reports
- **Artifact Management**: Saves model, scaler, and feature metadata with unique IDs
- **Detailed Logging**: Both detailed JSON logs and CSV history for easy tracking

### Process Flow:
```
Raw Data → Feature Analysis → Feature Selection → Preprocessing → Training → Evaluation → Artifacts → Logging
```

### Usage:
```python
from train_pipeline import MLTrainingPipeline

# Interactive mode
pipeline = MLTrainingPipeline()
result = pipeline.run_training_pipeline()

# Automated mode with specific features
result = pipeline.run_training_pipeline(feature_indices=[1, 2, 4, 5])
```

### Generated Artifacts:
- `model_YYYYMMDD_HHMMSS.pkl` - Trained RandomForest model
- `scaler_YYYYMMDD_HHMMSS.pkl` - StandardScaler fitted on training data
- `features_YYYYMMDD_HHMMSS.json` - Feature metadata and training configuration

### Generated Logs:
- `training_session_YYYYMMDD_HHMMSS.json` - Detailed session information
- `training_history.csv` - Master log of all training sessions

## 🔮 Pipeline 2: Prediction Pipeline

**File**: `predict_pipeline.py`

### Features:
- **Fully Automated**: No human intervention required
- **Latest Model Detection**: Automatically uses the most recent trained model
- **Data Validation**: Ensures incoming data matches training feature requirements
- **Preprocessing Consistency**: Uses the same preprocessing as training
- **Timestamped Output**: Each prediction run creates a unique output file

### Process Flow:
```
Incoming Data → Load Latest Artifacts → Preprocessing → Feature Scaling → Prediction → Output Packaging
```

### Usage:
```python
from predict_pipeline import MLPredictionPipeline

# Automated prediction
pipeline = MLPredictionPipeline()
result = pipeline.run_prediction_pipeline()
```

### Input Requirements:
- Incoming data must contain all features used during training
- Data should be in CSV format at `data/incoming_data.csv`

### Output Format:
- Original data columns + `prediction` + `probability_class_N` columns
- Saved as `prediction_YYYYMMDD_HHMMSS.csv` in output directory

## 🎛️ Pipeline Manager

**File**: `pipeline_manager.py`

### Features:
- **Unified Interface**: Single entry point for all pipeline operations
- **Interactive Menu**: Easy-to-use command-line interface
- **System Health Checks**: Monitor pipeline status and file integrity
- **Model Management**: View, compare, and manage trained models
- **Training History**: Track all training sessions and performance

### Main Menu Options:
1. **🔧 Training Pipeline** - Train a new model with feature selection
2. **🔮 Prediction Pipeline** - Make predictions with existing model
3. **📊 View Training History** - See all training sessions and scores
4. **🎯 Model Management** - Manage and inspect trained models
5. **📋 System Status** - Check system health and file integrity
6. **❌ Exit** - Close the application

### Usage:
```bash
python pipeline_manager.py
```

## 🗂️ Directory Structure

The pipelines expect the following directory structure:

```
project_root/
├── pipelines/                 # Pipeline code (this directory)
├── training/
│   ├── data/
│   │   └── raw_data.csv       # Training dataset
│   ├── artifacts/             # Model artifacts (auto-created)
│   └── logs/                  # Training logs (auto-created)
└── prediction/
    ├── data/
    │   └── incoming_data.csv   # Data to predict
    └── output/                 # Prediction results (auto-created)
```

## 📊 Training Logs Schema

### Master Training History (`training_history.csv`):
```csv
timestamp,training_id,n_features,features_list,train_size,test_size,train_score,test_score,cv_mean,cv_std,model_path,scaler_path,is_active
```

### Detailed Session Log (`training_session_YYYYMMDD_HHMMSS.json`):
```json
{
  "timestamp": "2025-07-02T14:30:00",
  "training_id": "20250702_143000",
  "features_used": ["feature1", "feature2", "..."],
  "n_features": 7,
  "train_size": 712,
  "test_size": 179,
  "train_score": 0.9944,
  "test_score": 0.8212,
  "cv_mean": 0.8156,
  "cv_std": 0.0234,
  "model_path": "artifacts/model_20250702_143000.pkl",
  "scaler_path": "artifacts/scaler_20250702_143000.pkl",
  "is_active": true
}
```

## 🎯 Key Features

### Training Pipeline Benefits:
1. **Interactive Feature Selection**: See feature statistics before selection
2. **Comprehensive Evaluation**: Cross-validation and detailed metrics
3. **Artifact Versioning**: Unique IDs for each training session
4. **History Tracking**: Complete log of all experiments
5. **Reproducibility**: All parameters and configurations saved

### Prediction Pipeline Benefits:
1. **Zero Configuration**: Automatically finds and uses latest model
2. **Data Validation**: Ensures data compatibility with trained model
3. **Consistent Processing**: Uses exact same preprocessing as training
4. **Batch Processing**: Handle multiple predictions efficiently
5. **Timestamped Output**: Track all prediction runs

### Management Benefits:
1. **Unified Interface**: Single point of control for all operations
2. **Model Comparison**: Easy comparison of different training runs
3. **System Monitoring**: Health checks and status reporting
4. **User-Friendly**: Intuitive menu-driven interface

## 🚀 Getting Started

1. **Setup**: Ensure your data is in the correct directories
2. **Training**: Run `python pipeline_manager.py` and select option 1
3. **Prediction**: Add new data to `prediction/data/incoming_data.csv` and select option 2
4. **Monitor**: Use options 3-5 to monitor and manage your models

## 🔄 Workflow Example

1. **Prepare Training Data**: Place your dataset in `training/data/raw_data.csv`
2. **Train Model**: Use the training pipeline to select features and train
3. **Review Results**: Check training history and model performance
4. **Make Predictions**: Add new data and run prediction pipeline
5. **Monitor**: Use system status to track pipeline health

This architecture provides a complete, production-ready ML pipeline system with clear separation of concerns, comprehensive logging, and easy management capabilities.
