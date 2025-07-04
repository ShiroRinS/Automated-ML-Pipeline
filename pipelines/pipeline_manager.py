#!/usr/bin/env python3
"""
ML Pipeline Manager
Coordinates training and prediction pipelines, manages models, and provides interface
"""
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import sys
import os

# Add pipelines directory to path for imports
sys.path.append(str(Path(__file__).parent))

from train_pipeline import MLTrainingPipeline, view_training_history
from predict_pipeline import MLPredictionPipeline

class MLPipelineManager:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.training_data_path = self.base_dir / "training" / "data" / "raw_data.csv"
        self.prediction_data_path = self.base_dir / "prediction" / "data" / "incoming_data.csv"
        self.artifacts_dir = self.base_dir / "training" / "artifacts"
        self.logs_dir = self.base_dir / "training" / "logs"
        self.output_dir = self.base_dir / "prediction" / "output"
        
    def show_main_menu(self):
        """Display main menu options"""
        print("\n" + "="*60)
        print("🚀 ML Pipeline Manager")
        print("="*60)
        print("1. 🔧 Training Pipeline - Train a new model")
        print("2. 🔮 Prediction Pipeline - Make predictions with existing model")
        print("3. 📊 View Training History")
        print("4. 🎯 Model Management")
        print("5. 📋 System Status")
        print("6. 🌐 Open Web Dashboard")
        print("7. ❌ Exit")
        print("="*60)
        
    def run_training_pipeline(self):
        """Execute the training pipeline"""
        print("\n🔧 Starting Training Pipeline...")
        
        # Check if training data exists
        if not self.training_data_path.exists():
            print(f"❌ Training data not found at: {self.training_data_path}")
            return False
            
        try:
            # Change to training directory for relative paths
            original_dir = os.getcwd()
            os.chdir(self.base_dir / "training")
            
            pipeline = MLTrainingPipeline()
            result = pipeline.run_training_pipeline()
            
            os.chdir(original_dir)
            
            # Handle cancelled training
            if result.get('cancelled', False):
                print("\n🔄 Training cancelled - returning to main menu")
                return True  # Return True to avoid error message
            
            return result['success']
            
        except Exception as e:
            print(f"❌ Training pipeline failed: {str(e)}")
            os.chdir(original_dir)
            return False
    
    def run_prediction_pipeline(self):
        """Execute the prediction pipeline"""
        print("\n🔮 Starting Prediction Pipeline...")
        
        # Check if prediction data exists
        if not self.prediction_data_path.exists():
            print(f"❌ Prediction data not found at: {self.prediction_data_path}")
            return False
        
        # Check if any trained models exist
        if not any(self.artifacts_dir.glob("model_*.pkl")):
            print("❌ No trained models found. Please run training pipeline first.")
            return False
            
        try:
            # Change to prediction directory for relative paths
            original_dir = os.getcwd()
            os.chdir(self.base_dir / "prediction")
            
            pipeline = MLPredictionPipeline(
                artifacts_dir="../training/artifacts"
            )
            result = pipeline.run_prediction_pipeline()
            
            os.chdir(original_dir)
            return result['success']
            
        except Exception as e:
            print(f"❌ Prediction pipeline failed: {str(e)}")
            os.chdir(original_dir)
            return False
    
    def view_training_history(self):
        """Display training history"""
        print("\n📊 Training History")
        print("-" * 50)
        
        # Change to training directory for relative paths
        original_dir = os.getcwd()
        os.chdir(self.base_dir / "training")
        
        try:
            df = view_training_history()
            if df is not None:
                print(f"\nTotal training sessions: {len(df)}")
                if len(df) > 0:
                    best_model = df.loc[df['test_score'].idxmax()]
                    print(f"Best performing model: {best_model['training_id']} (Score: {best_model['test_score']:.4f})")
        except Exception as e:
            print(f"Error viewing history: {str(e)}")
        finally:
            os.chdir(original_dir)
    
    def manage_models(self):
        """Model management interface"""
        print("\n🎯 Model Management")
        print("-" * 50)
        
        # List available models
        model_files = list(self.artifacts_dir.glob("model_*.pkl"))
        if not model_files:
            print("No trained models found.")
            return
        
        print(f"Found {len(model_files)} trained models:")
        for i, model_file in enumerate(sorted(model_files), 1):
            training_id = model_file.stem.split('_')[-1]
            features_file = self.artifacts_dir / f"features_{training_id}.json"
            
            if features_file.exists():
                with open(features_file, 'r') as f:
                    features_info = json.load(f)
                print(f"{i}. {training_id} - {features_info.get('n_features', 'N/A')} features")
            else:
                print(f"{i}. {training_id} - No feature info available")
        
        print("\nModel management options:")
        print("1. View detailed model information")
        print("2. Set active model for predictions")
        print("3. Delete old models")
        print("4. Back to main menu")
        
        while True:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                self._view_model_details(model_files)
                input("\nPress Enter to continue...")
            elif choice == "2":
                print("\n💡 Note: Prediction pipeline automatically uses the latest model")
                print("The most recently trained model is always used for predictions.")
                input("\nPress Enter to continue...")
            elif choice == "3":
                print("\n🚧 Model deletion feature coming soon!")
                print("For now, you can manually delete files from the artifacts directory.")
                input("\nPress Enter to continue...")
            elif choice == "4" or choice.lower() in ['back', 'b']:
                print("↩️  Returning to main menu...")
                break
            else:
                print("❌ Invalid choice. Please try again.")
        
    def _view_model_details(self, model_files):
        """View detailed information about a specific model"""
        print("\nSelect a model to view details:")
        for i, model_file in enumerate(sorted(model_files), 1):
            training_id = model_file.stem.split('_')[-1]
            print(f"{i}. {training_id}")
        
        try:
            choice = int(input("Enter model number: ")) - 1
            if 0 <= choice < len(model_files):
                model_file = sorted(model_files)[choice]
                training_id = model_file.stem.split('_')[-1]
                
                # Load detailed information
                detailed_log = self.logs_dir / f"training_session_{training_id}.json"
                if detailed_log.exists():
                    with open(detailed_log, 'r') as f:
                        details = json.load(f)
                    
                    print(f"\n📋 Model Details: {training_id}")
                    print(f"Timestamp: {details.get('timestamp', 'N/A')}")
                    print(f"Features: {details.get('n_features', 'N/A')} features")
                    print(f"Training Size: {details.get('train_size', 'N/A')}")
                    print(f"Test Size: {details.get('test_size', 'N/A')}")
                    print(f"Test Score: {details.get('test_score', 'N/A'):.4f}")
                    print(f"CV Score: {details.get('cv_mean', 'N/A'):.4f} (+/- {details.get('cv_std', 0) * 2:.4f})")
                    print(f"Features Used: {', '.join(details.get('features_used', []))}")
                else:
                    print("Detailed information not available for this model.")
        except (ValueError, IndexError):
            print("Invalid selection.")
    
    def show_system_status(self):
        """Display system status and health checks"""
        print("\n📋 System Status")
        print("-" * 50)
        
        # Check directories
        dirs_to_check = [
            ("Training Data", self.training_data_path.parent),
            ("Artifacts", self.artifacts_dir),
            ("Logs", self.logs_dir),
            ("Prediction Data", self.prediction_data_path.parent),
            ("Output", self.output_dir)
        ]
        
        for name, path in dirs_to_check:
            status = "✅ OK" if path.exists() else "❌ Missing"
            print(f"{name}: {status} ({path})")
        
        # Check files
        print("\nKey Files:")
        training_data_status = "✅ Found" if self.training_data_path.exists() else "❌ Missing"
        prediction_data_status = "✅ Found" if self.prediction_data_path.exists() else "❌ Missing"
        print(f"Training Data: {training_data_status}")
        print(f"Prediction Data: {prediction_data_status}")
        
        # Check models
        model_count = len(list(self.artifacts_dir.glob("model_*.pkl"))) if self.artifacts_dir.exists() else 0
        print(f"Trained Models: {model_count} available")
        
        # Check recent activity
        if self.logs_dir.exists():
            master_log = self.logs_dir / "training_history.csv"
            if master_log.exists():
                df = pd.read_csv(master_log)
                if len(df) > 0:
                    last_training = df.iloc[-1]['timestamp']
                    print(f"Last Training: {last_training}")
                else:
                    print("Last Training: Never")
            else:
                print("Last Training: Never")
        
        # Check prediction outputs
        if self.output_dir.exists():
            output_files = list(self.output_dir.glob("prediction_*.csv"))
            if output_files:
                latest_output = max(output_files, key=os.path.getctime)
                print(f"Latest Prediction: {latest_output.name}")
            else:
                print("Latest Prediction: None")
    
    def open_web_dashboard(self):
        """Open the web dashboard in browser"""
        import webbrowser
        import subprocess
        import time
        import platform
        
        print("\n🌐 Starting Web Dashboard...")
        print("📊 Dashboard URL: http://localhost:5000")
        
        web_app_path = self.base_dir / "web_ui" / "app.py"
        
        if not web_app_path.exists():
            print("❌ Web UI not found. Please ensure web_ui/app.py exists.")
            return
        
        try:
            print("🚀 Starting Flask server in new terminal...")
            
            # Get the absolute path to the web app
            abs_web_app_path = web_app_path.resolve()
            
            # Detect OS and open terminal accordingly
            system = platform.system().lower()
            
            if system == "windows":
                # For Windows - open PowerShell in new window
                cmd = f'start powershell -Command "cd \"{self.base_dir.resolve()}\"; python web_ui/app.py"'
                subprocess.run(cmd, shell=True)
                print("✅ New PowerShell window opened with Flask server!")
                
            elif system == "darwin":  # macOS
                # For macOS - open Terminal in new window
                script = f'tell app "Terminal" to do script "cd {self.base_dir.resolve()} && python web_ui/app.py"'
                subprocess.run(["osascript", "-e", script])
                print("✅ New Terminal window opened with Flask server!")
                
            elif system == "linux":
                # For Linux - try common terminal emulators
                terminals = ["gnome-terminal", "konsole", "xterm", "terminator"]
                
                for terminal in terminals:
                    try:
                        if terminal == "gnome-terminal":
                            subprocess.run([terminal, "--", "bash", "-c", f"cd {self.base_dir.resolve()} && python web_ui/app.py; exec bash"])
                        else:
                            subprocess.run([terminal, "-e", f"bash -c 'cd {self.base_dir.resolve()} && python web_ui/app.py; exec bash'"])
                        print(f"✅ New {terminal} window opened with Flask server!")
                        break
                    except FileNotFoundError:
                        continue
                else:
                    print("❌ No suitable terminal emulator found.")
                    raise Exception("Terminal not found")
            
            # Wait a moment for server to start
            print("⏳ Waiting for Flask server to start...")
            time.sleep(3)
            
            # Open browser
            print("🌐 Opening browser...")
            webbrowser.open('http://localhost:5000')
            print("\n✅ Dashboard should now be available in your browser!")
            print("📊 URL: http://localhost:5000")
            
            print("\n💡 Tips:")
            print("   • The Flask server will run in the new terminal window")
            print("   • Close the terminal window to stop the server")
            print("   • Refresh your browser if the page doesn't load immediately")
            
        except Exception as e:
            print(f"❌ Failed to automatically start server: {str(e)}")
            print("\n🔧 Manual fallback:")
            print("   1. Open a new terminal window")
            print(f"   2. Navigate to: {self.base_dir.resolve()}")
            print("   3. Run: python web_ui/app.py")
            print("   4. Open http://localhost:5000 in your browser")
            
            # Still try to open browser
            try:
                webbrowser.open('http://localhost:5000')
                print("\n🌐 Browser opened - start the server manually to access dashboard")
            except:
                pass
    
    def run(self):
        """Main interface loop"""
        while True:
            self.show_main_menu()
            choice = input("Enter your choice (1-7): ").strip()
            
            if choice == "1":
                self.run_training_pipeline()
            elif choice == "2":
                self.run_prediction_pipeline()
            elif choice == "3":
                self.view_training_history()
            elif choice == "4":
                self.manage_models()
            elif choice == "5":
                self.show_system_status()
            elif choice == "6":
                self.open_web_dashboard()
            elif choice == "7":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
            
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    manager = MLPipelineManager()
    manager.run()
