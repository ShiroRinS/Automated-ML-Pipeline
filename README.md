# Automated ML Pipeline

Automated Pipeline to help AI engineer life dealing with basic process easier.

A quick prototype for an automated machine learning pipeline with training and prediction components.

## Project Structure

```
ml_quick_prototype/
├── training/
│   ├── data/
│   │   └── raw_data.csv
│   ├── artifacts/
│   │   ├── model_YYYYMMDD.pkl
│   │   ├── scaler_YYYYMMDD.pkl
│   │   └── features_YYYYMMDD.json
│   ├── logs/
│   │   └── train_logs.csv
│   └── train.py
├── prediction/
│   ├── data/
│   │   └── incoming_data.csv
│   ├── output/
│   │   └── prediction_YYYYMMDD.csv
│   └── predict.py
├── requirements.txt
└── README.md
```

## Features

- **Training Pipeline**: Trains a RandomForest classifier with automatic artifact saving
- **Prediction Pipeline**: Loads trained models and makes predictions on new data
- **Logging**: Comprehensive training event logging
- **Artifact Management**: Automatic versioning of models, scalers, and feature metadata

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

### Training

```bash
cd training
python train.py
```

This will:
- Load data from `data/raw_data.csv`
- Train a RandomForest model
- Save model artifacts to `artifacts/`
- Log training events to `logs/train_logs.csv`

### Prediction

```bash
cd prediction
python predict.py
```

This will:
- Load the latest trained model
- Process data from `data/incoming_data.csv`
- Save predictions to `output/prediction_YYYYMMDD.csv`

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn

## License

This project is open source and available under the MIT License.
