#!/usr/bin/env python3
"""
ML Pipeline Web UI
Flask application for viewing training history, predictions, and model management
"""
from flask import Flask, render_template, send_from_directory, jsonify, request
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
import glob

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ml_pipeline_ui_secret_key'

# Base directory paths
BASE_DIR = Path(__file__).parent.parent
TRAINING_DIR = BASE_DIR / "training"
PREDICTION_DIR = BASE_DIR / "prediction"
ARTIFACTS_DIR = TRAINING_DIR / "artifacts"
LOGS_DIR = TRAINING_DIR / "logs"
OUTPUT_DIR = PREDICTION_DIR / "output"

@app.route('/')
def index():
    """Main dashboard"""
    try:
        # Get system statistics
        stats = get_system_stats()
        return render_template('index.html', stats=stats)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/predictions')
def predictions():
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
        
        return render_template('predictions.html', files=prediction_files)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/predictions/<filename>')
def view_prediction(filename):
    """View specific prediction results"""
    try:
        file_path = OUTPUT_DIR / filename
        if not file_path.exists():
            return render_template('error.html', error=f"Prediction file {filename} not found")
        
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
        
        return render_template('prediction_detail.html', 
                             filename=filename,
                             tables=[df_display.to_html(classes='table table-striped', escape=False)],
                             summary=summary)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/training-history')
def training_history():
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
        
        return render_template('training_history.html', history=history_data)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/training-detail/<training_id>')
def training_detail(training_id):
    """View detailed training session information"""
    try:
        detail_file = LOGS_DIR / f"training_session_{training_id}.json"
        if not detail_file.exists():
            return render_template('error.html', error=f"Training session {training_id} not found")
        
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
        
        return render_template('training_detail.html', 
                             training_id=training_id,
                             details=formatted_details,
                             raw_details=details)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/models')
def models():
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
        
        return render_template('models.html', models=model_info)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/system-status')
def system_status():
    """System health and status"""
    try:
        status = {
            'directories': check_directories(),
            'files': check_files(),
            'recent_activity': get_recent_activity()
        }
        return render_template('system_status.html', status=status)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/stats')
def api_stats():
    """API endpoint for dashboard statistics"""
    try:
        stats = get_system_stats()
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

def get_system_stats():
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
    print("📊 Dashboard will be available at: http://localhost:5000")
    print("🔧 Make sure you have run some training and prediction pipelines first!")
    app.run(debug=True, host='0.0.0.0', port=5000)
