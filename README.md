# Automated ML Pipeline

Automated Pipeline to help AI engineers streamline their workflow with an interactive web UI and automated ML processes.

This project provides a complete machine learning pipeline with training, prediction, and model management through a web interface.

## Project Structure

```
ml_quick_prototype/
├── pipelines/           # Pipeline orchestration
│   ├── train_pipeline.py
│   └── predict_pipeline.py
├── training/           
│   ├── data/           # Training data
│   │   └── raw_data.csv
│   ├── artifacts/      # Model artifacts
│   │   ├── model_YYYYMMDD.pkl
│   │   ├── scaler_YYYYMMDD.pkl
│   │   └── features_YYYYMMDD.json
│   ├── logs/          # Training history
│   │   └── training_history.csv
│   └── train.py
├── prediction/
│   ├── data/          # Input data
│   │   └── incoming_data.csv
│   ├── output/        # Predictions
│   │   └── prediction_YYYYMMDD.csv
│   └── predict.py
├── web_ui/            # Web interface
│   ├── templates/     # HTML templates
│   └── app.py        # Flask application
├── requirements.txt
└── README.md
```

## Features

- **Interactive Web UI**: Manage models and predictions through a user-friendly interface
- **Training Pipeline**: Trains a RandomForest classifier with automatic artifact saving
- **Prediction Pipeline**: Loads trained models and makes predictions on new data
- **Feature Selection**: Clickable interface for selecting and managing model features
- **Logging**: Comprehensive training event logging with performance tracking
- **Artifact Management**: Automatic versioning of models, scalers, and feature metadata
- **Dashboard**: Real-time overview of system status and recent activities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ShiroRinS/Automated-ML-Pipeline.git
cd Automated-ML-Pipeline
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web Interface

1. Start the web server:
```bash
python web_ui/app.py
```
Access the UI at: http://localhost:5000

### Training

1. Place your training data in `training/data/raw_data.csv`
2. Run the training pipeline:
```bash
python pipelines/train_pipeline.py
```

This will:
- Load and preprocess the training data
- Train a RandomForest model with cross-validation
- Save model artifacts to `artifacts/`
- Log training events and performance metrics

### Prediction

1. Place prediction data in `prediction/data/incoming_data.csv`
2. Run the prediction pipeline:
```bash
python pipelines/predict_pipeline.py
```

This will:
- Load the latest trained model
- Process new data using saved feature configurations
- Save predictions to `output/prediction_YYYYMMDD.csv`

### Feature Selection

1. Navigate to the Models page in the web UI
2. Click the "View Features" button for any model
3. Use the clickable interface to select/deselect features
4. Features will be automatically applied to new predictions

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn

## License

This project is proprietary and not open source.
