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

from pipelines.data_cleaning_page import DataCleaningPage
from pipelines.model_training_page import ModelTrainingPage
from pipelines.feature_recommender import FeatureRecommender
from config import configure_gemini, ENABLE_GEMINI

app = Quart(__name__)

# Initialize pages and components
data_cleaning = DataCleaningPage()
model_training = ModelTrainingPage()
feature_recommender = FeatureRecommender()
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
        
        # Load initial data
        data_path = TRAINING_DIR / 'data' / 'titanic.csv'
        print(f"Looking for data at: {data_path}")
        print(f"Data file exists: {data_path.exists()}")
        
        try:
            # Load data
            data = pd.read_csv(data_path)
            print("Data loaded successfully")
            
            # Get feature recommendations
            print("Getting feature recommendations...")
            recommendations = await feature_recommender.get_recommendations(data)
            print("Got feature recommendations")
            print(f"Feature importance scores: {recommendations.get('feature_importances', [])}")
            
            # Ensure recommendations is serializable
            safe_recommendations = {
                'raw_suggestions': recommendations.get('gemini_suggestions', ''),
                'feature_importances': recommendations.get('feature_importances', []),
                'timestamp': str(recommendations.get('timestamp', ''))
            }
            print(f"Safe recommendations being passed to template: {safe_recommendations}")
            
            return await render_template('model_training.html', 
                                    initial_recommendations=safe_recommendations,
                                    data_loaded=True)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            return await render_template('model_training.html',
                                    error=error_msg,
                                    data_loaded=False)
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        return await render_template('model_training.html',
                                error=error_msg,
                                data_loaded=False)
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
            
        # Save file temporarily
        temp_path = TRAINING_DIR / 'data' / 'temp.csv'
        await file.save(temp_path)
        
        # Load and analyze data
        analysis = data_cleaning.load_data(str(temp_path))
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'columns': [
                {
                    'name': col,
                    'type': analysis['data_types'][col],
                    'missing_pct': analysis['missing_stats'][col]['percentage']
                }
                for col in analysis['data_types'].keys()
            ]
        })
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
        model_training.load_data(str(TRAINING_DIR / 'data' / 'cleaned_data.csv'))
        recommendations = await model_training.get_feature_recommendations()
        return jsonify({'success': True, **recommendations})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/train-model', methods=['POST'])
async def start_model_training():
    """Start model training with selected features"""
    try:
        data = await request.get_json()
        
        # Initialize pipeline
        from pipelines.train_pipeline import MLTrainingPipeline
        pipeline = MLTrainingPipeline()
        
        # Load and prepare data
        data_path = TRAINING_DIR / 'data' / 'titanic.csv'
        load_result = pipeline.load_data()
        
        if load_result is None:
            return jsonify({
                'success': False,
                'error': 'Failed to load training data'
            })
        
        # Convert feature names to indices
        feature_indices = []
        all_features = pipeline.features
        selected_features = data.get('features', [])
        
        for feature in selected_features:
            try:
                idx = all_features.index(feature) + 1  # 1-based index
                feature_indices.append(idx)
            except ValueError:
                continue
        
        if not feature_indices:
            return jsonify({
                'success': False,
                'error': 'No valid features selected'
            })
        
        # Run training pipeline
        result = pipeline.run_training_pipeline(feature_indices=feature_indices)
        
        if result['success']:
            return jsonify({
                'success': True,
                'results': {
                    'training_id': result['training_id'],
                    'metrics': result['results'],
                    'model_path': result['artifacts']['model_path'],
                    'feature_importance': result['log_entry']['features_used']
                }
            })
        else:
            error_msg = result.get('error', 'Unknown error occurred')
            return jsonify({
                'success': False,
                'error': error_msg
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
        
        # Add human-readable interpretations
        df_display = df.copy()
        if 'Sex' in df_display.columns:
            df_display['Gender'] = df_display['Sex'].map({0: 'Female', 1: 'Male'})
        if 'Pclass' in df_display.columns:
            df_display['Class'] = df_display['Pclass'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})
        if 'Embarked' in df_display.columns:
            df_display['Port'] = df_display['Embarked'].map({0: 'Cherbourg', 1: 'Queenstown', 2: 'Southampton'})
        if 'prediction' in df_display.columns:
            df_display['Survival'] = df_display['prediction'].map({0: 'Did Not Survive', 1: 'Survived'})
        
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
        data_path = TRAINING_DIR / "data" / "titanic.csv"
        df = pd.read_csv(data_path)
        features = list(df.columns[:-1])  # Assume last column is target
        print(f"Test endpoint features: {features}")
        return jsonify({
            'success': True,
            'message': 'API is working',
            'features': features
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/training/features')
def get_available_features():
    """Get available features from training data"""
    print("\n=== Feature Loading Debug Log ===")
    print("Request received for features")
    try:
        print("1. Loading features...")
        # Load data only once and cache it
        data_path = TRAINING_DIR / "data" / "titanic.csv"
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
        
        features = list(df.columns[:-1])  # Assume last column is target
        print(f"7. Extracted features: {features}")
        
        print("8. Analyzing features...")
        feature_info = []
        try:
            for i, feature in enumerate(features):
                print(f"   Processing feature: {feature}")
                feature_data = df[feature]
                missing_count = feature_data.isnull().sum()
                missing_pct = (missing_count / len(feature_data)) * 100
                
                info = {
                    'index': i + 1,
                    'name': feature,
                    'type': str(feature_data.dtype),
                    'missing': int(missing_count),
                    'missing_pct': float(missing_pct),
                    'unique_values': int(feature_data.nunique()),
                    'sample_values': feature_data.dropna().head().tolist()
                }
                
                if feature_data.dtype in ['int64', 'float64']:
                    info.update({
                        'mean': float(feature_data.mean()),
                        'std': float(feature_data.std()),
                        'min': float(feature_data.min()),
                        'max': float(feature_data.max())
                    })
                
                feature_info.append(info)
                print(f"   Completed feature: {feature}")
        except Exception as e:
            print(f"ERROR analyzing features: {str(e)}")
            raise
        
        print("9. Preparing JSON response...")
        response = {
            'success': True,
            'features': feature_info,
            'target': df.columns[-1]
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
