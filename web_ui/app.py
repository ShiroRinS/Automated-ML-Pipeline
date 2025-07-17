#!/usr/bin/env python3
"""
ML Pipeline Web UI
Flask application for viewing training history, predictions, and model management
"""
from quart import Quart, render_template, send_from_directory, jsonify, request
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
import glob
import sys

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from pipelines.data_cleaning_enhanced import DataCleaningEnhanced
from pipelines.model_training_page import ModelTrainingPage
from pipelines.feature_recommender import FeatureRecommender
from pipelines.data_preprocessing import DataPreprocessor
from config import configure_gemini, ENABLE_GEMINI

app = Quart(__name__)

# Initialize pages and components
data_cleaning = DataCleaningEnhanced()
model_training = ModelTrainingPage()
feature_recommender = FeatureRecommender()
preprocessor = DataPreprocessor()
app.config['SECRET_KEY'] = 'ml_pipeline_ui_secret_key'

# Base directory paths
BASE_DIR = Path(__file__).parent.parent
TRAINING_DIR = BASE_DIR / "training"
PREDICTION_DIR = BASE_DIR / "prediction"
ARTIFACTS_DIR = TRAINING_DIR / "artifacts"
LOGS_DIR = TRAINING_DIR / "logs"
OUTPUT_DIR = PREDICTION_DIR / "output"

@app.route('/')
async def index():
    """Main dashboard"""
    try:
        # Get system statistics
        stats = await get_system_stats()
        return await render_template('index.html', stats=stats)
    except Exception as e:
        return await render_template('error.html', error=str(e))

@app.route('/clean-data')
async def clean_data():
    """Data cleaning page"""
    try:
        return await render_template('data_cleaning.html')
    except Exception as e:
        return await render_template('error.html', error=str(e))

@app.route('/train')
async def train_model():
    """Train new model with feature selection"""
    try:
        print("Loading training page...")
        
        # Load data from the designated data directory
        data_path = TRAINING_DIR / "data"
        csv_files = list(data_path.glob("*.csv"))
        
        if not csv_files:
            print("No CSV files found in training data directory")
            return await render_template('training.html',
                                error="No CSV files found in the training data directory.",
                                data_loaded=False,
                                data_analysis=None,
                                initial_recommendations=None)
        
        # Use the most recently modified CSV file
        data_path = max(csv_files, key=lambda x: x.stat().st_mtime)
        print(f"Looking for data at: {data_path}")
        print(f"Data file exists: {data_path.exists()}")
        
        try:
            # Load data
            data = pd.read_csv(data_path)
            print("Data loaded successfully")
            print(f"Data shape: {data.shape}")
            print(f"Columns: {data.columns.tolist()}")
            
            # Get column information
            column_info = {
                'numeric': data.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                'categorical': data.select_dtypes(include=['object', 'category']).columns.tolist(),
                'excluded': [col for col in data.columns if any(text in col.lower() for text in ['id', 'name', 'ticket', 'phone', 'email', 'address', 'description'])]
            }
            
            # Initialize data analysis
            data_analysis = {
                'total_rows': int(len(data)),
                'total_columns': int(len(data.columns)),
                'column_info': column_info,
                'missing_values_total': int(data.isnull().sum().sum())
            }
            
            # Analyze data columns and types
            print("Analyzing data structure...")
            print(f"Numeric columns: {column_info['numeric']}")
            print(f"Categorical columns: {column_info['categorical']}")
            print(f"Excluded columns: {column_info['excluded']}")
            
            # Get basic statistics for numeric columns
            numeric_stats = {}
            for col in column_info['numeric']:
                col_data = data[col]
                numeric_stats[col] = {
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'missing': int(col_data.isna().sum()),
                    'missing_pct': float(col_data.isna().mean() * 100)
                    
                }
            
            # Get value counts for categorical columns
            categorical_stats = {}
            for col in column_info['categorical']:
                col_data = data[col]
                value_counts = col_data.value_counts(normalize=True)
                categorical_stats[col] = {
                    'unique_values': int(col_data.nunique()),
                    'missing': int(col_data.isna().sum()),
                    'missing_pct': float(col_data.isna().mean() * 100),
                    'most_common': value_counts.index[0] if not value_counts.empty else None,
                    'most_common_pct': float(value_counts.iloc[0] * 100) if not value_counts.empty else 0,
                }
            
            # Add statistics to data analysis
            data_analysis['numeric_stats'] = numeric_stats
            data_analysis['categorical_stats'] = categorical_stats
            
            # Add timestamp to data analysis
            data_analysis['timestamp'] = str(pd.Timestamp.now())
            
            # Debug: Print final data_analysis
            print("\nFinal data_analysis structure:")
            print(f"Keys: {data_analysis.keys()}")
            print(f"Column info keys: {data_analysis['column_info'].keys()}")
            print(f"Has numeric stats: {bool(data_analysis['numeric_stats'])}")
            print(f"Has categorical stats: {bool(data_analysis['categorical_stats'])}")
            
            # Prepare initial recommendations (empty for now)
            initial_recommendations = {
                'feature_importances': [],
                'raw_suggestions': None
            }
            
            print("\nPreparing to render template with:")
            print(f"data_loaded=True")
            print(f"data_analysis keys: {data_analysis.keys()}")
            print(f"initial_recommendations: {initial_recommendations}")
            
            return await render_template('training.html',
                                    data_loaded=True,
                                    data_analysis=data_analysis,
                                    initial_recommendations=initial_recommendations)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            return await render_template('training.html',
                                    error=error_msg,
                                    data_loaded=False,
                                    data_analysis=None,
                                    initial_recommendations=None)
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        return await render_template('training.html',
                                error=error_msg,
                                data_loaded=False,
                                data_analysis=None,
                                initial_recommendations=None)
    except Exception as e:
        print(f"Error in train_model route: {str(e)}")
        return await render_template('error.html', error=str(e))

@app.route('/api/upload-data', methods=['POST'])
async def upload_data():
    """Handle data file upload"""
    try:
        files = await request.files
        if 'file' not in files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
            
        file = files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
            
        # Ensure directory exists
        data_dir = TRAINING_DIR / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
            
        # Save file temporarily
        temp_path = data_dir / 'temp.csv'
        await file.save(temp_path)
        
        try:
            # Load and analyze data
            analysis = data_cleaning.load_data(str(temp_path))
            
            # Extract column information
            columns = []
            for col_info in analysis['columns']:
                columns.append({
                    'name': col_info['name'],
                    'type': col_info['type'],
                    'missing_pct': col_info['missing_pct']
                })
            
            return jsonify({
                'success': True,
                'analysis': {
                    'total_rows': analysis['total_rows'],
                    'total_columns': analysis['total_columns'],
                    'quality_score': analysis['quality_score'],
                    'data_types': {col['name']: col['type'] for col in analysis['columns']},
                    'missing_stats': {col['name']: {'percentage': col['missing_pct']} for col in analysis['columns']}
                },
                'columns': columns
            })
        except Exception as e:
            return jsonify({'success': False, 'error': f'Error analyzing data: {str(e)}'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error uploading file: {str(e)}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/cleaning-options/<column>')
async def get_cleaning_options(column):
    """Get cleaning options for a column"""
    try:
        options = data_cleaning.get_cleaning_options()[column]
        return jsonify({'success': True, 'options': options})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/apply-cleaning', methods=['POST'])
async def apply_cleaning():
    """Apply cleaning method to data"""
    try:
        data = await request.get_json()
        df, message = data_cleaning.apply_cleaning(
            column=data['column'],
            method=data['method'],
            params=data.get('params')
        )
        
        # Generate preview HTML
        preview = df.head().to_html(classes='table table-striped')
        
        return jsonify({
            'success': True,
            'preview': preview,
            'history': data_cleaning.get_cleaning_history(),
            'message': message
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/prepare-data', methods=['POST'])
async def prepare_data():
    """Prepare data with selected actions like normalization or scaling"""
    try:
        data = await request.get_json()
        action = data.get('action')

        # Use methods from DataCleaningEnhanced or new utility class
        if action == 'normalize':
            df = data_cleaning.normalize_data()
            message = 'Data normalized successfully'
        elif action == 'scale':
            df = data_cleaning.scale_features()
            message = 'Features scaled successfully'
        else:
            return jsonify({'success': False, 'error': 'Invalid action'})

        # Generate preview HTML
        preview = df.head().to_html(classes='table table-striped')
        
        return jsonify({
            'success': True,
            'preview': preview,
            'message': message
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/undo-cleaning', methods=['POST'])
async def undo_cleaning():
    """Undo last cleaning operation"""
    try:
        df, message = data_cleaning.undo_last_cleaning()
        preview = df.head().to_html(classes='table table-striped')
        
        return jsonify({
            'success': True,
            'preview': preview,
            'history': data_cleaning.get_cleaning_history(),
            'message': message
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/save-cleaned-data', methods=['POST'])
async def save_cleaned_data():
    """Save cleaned data"""
    try:
        output_path = TRAINING_DIR / 'data' / 'cleaned_data.csv'
        message = data_cleaning.save_cleaned_data(str(output_path))
        return jsonify({'success': True, 'message': message})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/feature-recommendations')
async def get_feature_recommendations():
    """Get feature recommendations from Gemini"""
    try:
        # Find most recent data file
        data_path = TRAINING_DIR / "data"
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            return jsonify({
                'success': False,
                'error': 'No training data found'
            })
            
        data_path = max(csv_files, key=lambda x: x.stat().st_mtime)
        
        # Load data and get recommendations
        data = pd.read_csv(data_path)
        recommendations = await feature_recommender.get_recommendations(data)
        
        # Enhance recommendations with feature types
        if 'feature_importances' in recommendations:
            numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
            for feature in recommendations['feature_importances']:
                feature['type'] = 'numeric' if feature['feature'] in numeric_features else 'categorical'
        
        return jsonify({'success': True, **recommendations})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/train-model', methods=['POST'])
async def start_model_training():
    """Start model training with selected features and preprocessing options"""
    try:
        data = await request.get_json()
        selected_features = data.get('features', [])
        test_size = data.get('test_size', 0.2)
        categorical_encoding = data.get('categorical_encoding', {})
        excluded_columns = data.get('excluded_columns', [])
        target_column = data.get('target_column', None)  # Allow user to specify target
        
        # Find most recent data file
        data_path = TRAINING_DIR / 'data'
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            return jsonify({
                'success': False,
                'error': 'No training data found'
            })
            
        data_path = max(csv_files, key=lambda x: x.stat().st_mtime)
        
        # Load and preprocess data
        preprocessor.load_data(data_path)
        
        # Configure preprocessing based on feature types
        preprocessing_choices = {
            'missing_actions': {},
            'encoding': categorical_encoding,
            'scaling': 'standard'
        }
        
        # Set default missing value handling based on data type
        for feature in selected_features:
            if feature in preprocessor.data.select_dtypes(include=['int64', 'float64']).columns:
                preprocessing_choices['missing_actions'][feature] = 'mean'
            else:
                preprocessing_choices['missing_actions'][feature] = 'mode'
        
        preprocessed_data, summary = preprocessor.preprocess_data(preprocessing_choices)
        
        # Train model with preprocessed data (use last column as target by default)
        target_variable = preprocessor.data.columns[-1]
        model_training.load_data(data_path, target_variable)
        result = model_training.train_initial_model(selected_features, test_size)
        
        return jsonify({
            'success': True,
            'results': result,
            'preprocessing_summary': summary
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/tune-model', methods=['POST'])
async def tune_model():
    """Tune model hyperparameters"""
    try:
        params = await request.get_json()
        results = model_training.tune_model(param_grid=params)
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/save-model', methods=['POST'])
async def save_trained_model():
    """Save trained model"""
    try:
        paths = model_training.save_model(str(ARTIFACTS_DIR))
        return jsonify({'success': True, 'paths': paths})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predictions')
async def predictions():
    """View all prediction results"""
    try:
        prediction_files = []
        if OUTPUT_DIR.exists():
            for file in sorted(OUTPUT_DIR.glob("prediction_*.csv"), reverse=True):
                file_stats = file.stat()
                prediction_files.append({
                    'filename': file.name,
                    'size': f"{file_stats.st_size / 1024:.1f} KB",
                    'modified': datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    'path': str(file.relative_to(BASE_DIR))
                })
        
        return await render_template('predictions.html', files=prediction_files)
    except Exception as e:
        return await render_template('error.html', error=str(e))

@app.route('/predictions/<filename>')
async def view_prediction(filename):
    """View specific prediction results"""
    try:
        file_path = OUTPUT_DIR / filename
        if not file_path.exists():
            return await render_template('error.html', error=f"Prediction file {filename} not found")
        
        df = pd.read_csv(file_path)
        df_display = df.copy()
        
        # Add generic prediction label if it exists (assuming binary classification)
        if 'prediction' in df_display.columns:
            unique_vals = sorted(df_display['prediction'].unique())
            if len(unique_vals) == 2:
                df_display['Prediction'] = df_display['prediction'].map({min(unique_vals): 'Negative', max(unique_vals): 'Positive'})
        
        # Summary statistics
        summary = {}
        if 'prediction' in df.columns:
            summary['total_predictions'] = len(df)
            summary['survived'] = int(df['prediction'].sum())
            summary['not_survived'] = len(df) - summary['survived']
            summary['survival_rate'] = f"{(summary['survived'] / len(df)) * 100:.1f}%"
        
        return await render_template('prediction_detail.html', 
                             filename=filename,
                             tables=[df_display.to_html(classes='table table-striped', escape=False)],
                             summary=summary)
    except Exception as e:
        return await render_template('error.html', error=str(e))

@app.route('/training-history')
async def training_history():
    """View training history"""
    try:
        history_data = []
        master_log = LOGS_DIR / "training_history.csv"
        
        if master_log.exists():
            df = pd.read_csv(master_log)
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            df['test_score'] = df['test_score'].round(4)
            df['cv_mean'] = df['cv_mean'].round(4)
            df['cv_std'] = df['cv_std'].round(4)
            
            history_data = df.to_dict('records')
        
        return await render_template('training_history.html', history=history_data)
    except Exception as e:
        return await render_template('error.html', error=str(e))

@app.route('/training-detail/<training_id>')
async def training_detail(training_id):
    """View detailed training session information"""
    try:
        detail_file = LOGS_DIR / f"training_session_{training_id}.json"
        if not detail_file.exists():
            return await render_template('error.html', error=f"Training session {training_id} not found")
        
        with open(detail_file, 'r') as f:
            details = json.load(f)
        
        # Format the details for display
        formatted_details = {
            'Training ID': details.get('training_id', 'N/A'),
            'Timestamp': details.get('timestamp', 'N/A'),
            'Features Used': len(details.get('features_used', [])),
            'Feature List': ', '.join(details.get('features_used', [])),
            'Training Size': details.get('train_size', 'N/A'),
            'Test Size': details.get('test_size', 'N/A'),
            'Total Data Size': details.get('total_data_size', 'N/A'),
            'Training Score': f"{details.get('train_score', 0):.4f}",
            'Test Score': f"{details.get('test_score', 0):.4f}",
            'CV Mean': f"{details.get('cv_mean', 0):.4f}",
            'CV Std': f"{details.get('cv_std', 0):.4f}",
            'Model Path': details.get('model_path', 'N/A'),
            'Scaler Path': details.get('scaler_path', 'N/A'),
            'Is Active': 'Yes' if details.get('is_active', False) else 'No',
            'Notes': details.get('notes', 'N/A')
        }
        
        return await render_template('training_detail.html', 
                             training_id=training_id,
                             details=formatted_details,
                             raw_details=details)
    except Exception as e:
        return await render_template('error.html', error=str(e))

@app.route('/models')
async def models():
    """View available models"""
    try:
        model_info = []
        if ARTIFACTS_DIR.exists():
            model_files = list(ARTIFACTS_DIR.glob("model_*.pkl"))
            for model_file in sorted(model_files, reverse=True):
                training_id = '_'.join(model_file.stem.split('_')[1:])
                features_file = ARTIFACTS_DIR / f"features_{training_id}.json"
                
                info = {
                    'training_id': training_id,
                    'model_file': model_file.name,
                    'size': f"{model_file.stat().st_size / 1024:.1f} KB",
                    'modified': datetime.fromtimestamp(model_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    'features': 'N/A',
                    'target': 'N/A'
                }
                
                if features_file.exists():
                    with open(features_file, 'r') as f:
                        features_data = json.load(f)
                        info['features'] = features_data.get('n_features', 'N/A')
                        info['target'] = features_data.get('target_variable', 'N/A')
                
                model_info.append(info)
        
        return await render_template('models.html', models=model_info)
    except Exception as e:
        return await render_template('error.html', error=str(e))

@app.route('/system-status')
async def system_status():
    """System health and status"""
    try:
        status = {
            'directories': check_directories(),
            'files': check_files(),
            'recent_activity': get_recent_activity()
        }
        return await render_template('system_status.html', status=status)
    except Exception as e:
        return await render_template('error.html', error=str(e))

@app.route('/config')
async def config_page():
    """Configuration page"""
    try:
        return await render_template('config.html')
    except Exception as e:
        return await render_template('error.html', error=str(e))

@app.route('/api/configure', methods=['POST'])
async def configure_api():
    """Configure API keys"""
    try:
        data = await request.get_json()
        gemini_key = data.get('gemini_api_key')
        
        if not gemini_key:
            return jsonify({
                'success': False,
                'error': 'Gemini API key is required'
            })
        
        # Configure Gemini
        try:
            configure_gemini(gemini_key)
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Error configuring Gemini: {str(e)}'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/get-gemini-suggestions')
async def get_gemini_suggestions():
    """Get feature recommendations from Gemini"""
    try:
        # Find the most recent CSV file
        data_path = TRAINING_DIR / "data"
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            return jsonify({
                'success': False,
                'error': 'No training data found'
            })
            
        data_path = max(csv_files, key=lambda x: x.stat().st_mtime)
        data = pd.read_csv(data_path)
        
        # Get recommendations
        recommendations = await feature_recommender.get_recommendations(data)
        if not recommendations or 'gemini_suggestions' not in recommendations:
            return jsonify({
                'success': False,
                'error': 'Failed to get AI suggestions'
            })
            
        return jsonify({
            'success': True,
            'suggestions': recommendations['gemini_suggestions']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/config-status')
async def config_status():
    """Get configuration status"""
    return jsonify({
        'gemini_enabled': ENABLE_GEMINI
    })

@app.route('/api/stats')
async def api_stats():
    """API endpoint for dashboard statistics"""
    try:
        stats = await get_system_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/prediction/<filename>')
def download_prediction(filename):
    """Download prediction file"""
    try:
        return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/test')
def test_endpoint():
    """Test endpoint to verify API is working and return features"""
    print("Test endpoint called")
    try:
        # Find most recent data file
        data_path = TRAINING_DIR / "data"
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            return jsonify({
                'success': False,
                'error': 'No training data found'
            })
            
        data_path = max(csv_files, key=lambda x: x.stat().st_mtime)
        df = pd.read_csv(data_path)
        
        # Get feature types
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target (last column) from feature lists
        if len(df.columns) > 0:
            if df.columns[-1] in numeric_features:
                numeric_features.remove(df.columns[-1])
            if df.columns[-1] in categorical_features:
                categorical_features.remove(df.columns[-1])
        
        print(f"Test endpoint features - Numeric: {numeric_features}, Categorical: {categorical_features}")
        return jsonify({
            'success': True,
            'message': 'API is working',
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'target': df.columns[-1] if len(df.columns) > 0 else None
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/training/features')
def get_available_features():
    """Get available features from training data with enhanced analysis"""
    print("\n=== Feature Loading Debug Log ===")
    print("Request received for features")
    try:
        print("1. Loading features...")
        # Find the most recent CSV file in the data directory
        data_path = TRAINING_DIR / "data"
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV files found in training data directory")
        data_path = max(csv_files, key=lambda x: x.stat().st_mtime)
        print(f"2. Data path: {data_path}")
        print(f"3. Data path exists: {data_path.exists()}")
        
        if not data_path.exists():
            print("ERROR: Data file not found!")
            raise FileNotFoundError(f"Training data not found at {data_path}")
        
        print("4. Reading CSV file...")
        try:
            df = pd.read_csv(data_path)
            print(f"5. CSV loaded successfully. Shape: {df.shape}")
            print(f"6. Columns: {list(df.columns)}")
        except Exception as e:
            print(f"ERROR reading CSV: {str(e)}")
            raise
        
        # Identify feature types
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Get target variable (last column)
        target = df.columns[-1] if len(df.columns) > 0 else None
        
        # Remove target from feature lists if present
        if target in numeric_features:
            numeric_features.remove(target)
        if target in categorical_features:
            categorical_features.remove(target)
            
        all_features = numeric_features + categorical_features
        print(f"7. Extracted features - Numeric: {numeric_features}, Categorical: {categorical_features}")
        
        print("8. Analyzing features...")
        feature_info = []
        try:
            for i, feature in enumerate(all_features):
                print(f"   Processing feature: {feature}")
                feature_data = df[feature]
                missing_count = feature_data.isnull().sum()
                missing_pct = (missing_count / len(feature_data)) * 100
                is_numeric = feature in numeric_features
                
                info = {
                    'index': i + 1,
                    'name': feature,
                    'type': 'numeric' if is_numeric else 'categorical',
                    'data_type': str(feature_data.dtype),
                    'missing': int(missing_count),
                    'missing_pct': float(missing_pct),
                    'unique_values': int(feature_data.nunique()),
                    'sample_values': feature_data.dropna().head().tolist(),
                    'preprocessing_options': {
                        'missing_handling': ['mean', 'median', 'mode', 'constant'] if is_numeric else ['mode', 'constant', 'unknown'],
                        'encoding': [] if is_numeric else ['label', 'onehot', 'exclude'],
                        'scaling': ['standard', 'minmax', 'robust', 'none'] if is_numeric else []
                    }
                }
                
                if is_numeric:
                    info.update({
                        'statistics': {
                            'mean': float(feature_data.mean()),
                            'std': float(feature_data.std()),
                            'min': float(feature_data.min()),
                            'max': float(feature_data.max()),
                            'median': float(feature_data.median()),
                            'q1': float(feature_data.quantile(0.25)),
                            'q3': float(feature_data.quantile(0.75))
                        }
                    })
                else:
                    # Add categorical statistics
                    value_counts = feature_data.value_counts(normalize=True)
                    info.update({
                        'statistics': {
                            'mode': feature_data.mode()[0],
                            'unique_count': len(value_counts),
                            'most_frequent': value_counts.index[0],
                            'most_frequent_pct': float(value_counts.iloc[0] * 100),
                            'value_distribution': {
                                str(k): float(v) for k, v in value_counts.head().items()
                            }
                        }
                    })
                
                feature_info.append(info)
                print(f"   Completed feature: {feature}")
        except Exception as e:
            print(f"ERROR analyzing features: {str(e)}")
            raise
        
        # Analyze target variable
        target_info = None
        if target:
            target_data = df[target]
            is_numeric_target = target in df.select_dtypes(include=['int64', 'float64']).columns
            
            target_info = {
                'name': target,
                'type': 'numeric' if is_numeric_target else 'categorical',
                'data_type': str(target_data.dtype),
                'unique_values': int(target_data.nunique())
            }
            
            if is_numeric_target:
                target_info.update({
                    'statistics': {
                        'mean': float(target_data.mean()),
                        'std': float(target_data.std()),
                        'min': float(target_data.min()),
                        'max': float(target_data.max())
                    }
                })
            else:
                value_counts = target_data.value_counts(normalize=True)
                target_info.update({
                    'statistics': {
                        'classes': value_counts.index.tolist(),
                        'class_distribution': {
                            str(k): float(v) for k, v in value_counts.items()
                        }
                    }
                })
        
        print("9. Preparing JSON response...")
        response = {
            'success': True,
            'dataset_info': {
                'total_samples': len(df),
                'total_features': len(all_features),
                'numeric_features': len(numeric_features),
                'categorical_features': len(categorical_features),
                'missing_values_total': df.isnull().sum().sum(),
                'file_name': data_path.name,
                'file_timestamp': datetime.fromtimestamp(data_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            },
            'features': feature_info,
            'target': target_info
        }
        print("10. Returning response")
        return jsonify(response)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/training/start', methods=['POST'])
def start_training():
    """Start training with selected features"""
    try:
        feature_indices = request.json.get('features', [])
        
        from pipelines.train_pipeline import MLTrainingPipeline
        pipeline = MLTrainingPipeline()
        result = pipeline.run_training_pipeline(feature_indices=feature_indices)
        
        return jsonify({
            'success': True,
            'result': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

async def get_system_stats():
    """Get system statistics for dashboard"""
    stats = {
        'total_models': 0,
        'total_predictions': 0,
        'training_sessions': 0,
        'latest_model': 'None',
        'latest_prediction': 'None',
        'best_score': 0.0
    }
    
    # Count models
    if ARTIFACTS_DIR.exists():
        stats['total_models'] = len(list(ARTIFACTS_DIR.glob("model_*.pkl")))
    
    # Count predictions
    if OUTPUT_DIR.exists():
        prediction_files = list(OUTPUT_DIR.glob("prediction_*.csv"))
        stats['total_predictions'] = len(prediction_files)
        if prediction_files:
            latest_pred = max(prediction_files, key=os.path.getctime)
            stats['latest_prediction'] = latest_pred.name
    
    # Training history
    master_log = LOGS_DIR / "training_history.csv"
    if master_log.exists():
        df = pd.read_csv(master_log)
        stats['training_sessions'] = len(df)
        if len(df) > 0:
            best_model = df.loc[df['test_score'].idxmax()]
            stats['best_score'] = f"{best_model['test_score']:.4f}"
            stats['latest_model'] = best_model['training_id']
    
    # Add recent activity
    stats['recent_activity'] = get_recent_activity()
    
    return stats

def check_directories():
    """Check directory status"""
    dirs = {
        'Training Data': TRAINING_DIR / "data",
        'Artifacts': ARTIFACTS_DIR,
        'Logs': LOGS_DIR,
        'Prediction Data': PREDICTION_DIR / "data",
        'Output': OUTPUT_DIR
    }
    
    return {name: path.exists() for name, path in dirs.items()}

def check_files():
    """Check key files status"""
    files = {
        'Training Data': (TRAINING_DIR / "data" / "raw_data.csv").exists(),
        'Prediction Data': (PREDICTION_DIR / "data" / "incoming_data.csv").exists(),
        'Training History': (LOGS_DIR / "training_history.csv").exists()
    }
    
    return files

def get_recent_activity():
    """Get recent system activity"""
    activity = []
    
    # Recent predictions
    if OUTPUT_DIR.exists():
        for file in sorted(OUTPUT_DIR.glob("prediction_*.csv"), reverse=True)[:3]:
            activity.append({
                'type': 'Prediction',
                'description': f"Generated {file.name}",
                'timestamp': datetime.fromtimestamp(file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            })
    
    # Recent training
    master_log = LOGS_DIR / "training_history.csv"
    if master_log.exists():
        df = pd.read_csv(master_log)
        if len(df) > 0:
            for _, row in df.tail(3).iterrows():
                activity.append({
                    'type': 'Training',
                    'description': f"Trained model {row['training_id']} (Score: {row['test_score']:.4f})",
                    'timestamp': pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                })
    
    return sorted(activity, key=lambda x: x['timestamp'], reverse=True)[:5]

if __name__ == '__main__':
    print("🌐 Starting ML Pipeline Web UI...")
    print("📊 Dashboard will be available at: http://localhost:5001")
    print("🔧 Make sure you have run some training and prediction pipelines first!")
    app.run(debug=True, host='0.0.0.0', port=5001)
