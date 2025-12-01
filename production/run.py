#!/usr/bin/env python
"""
Simple Python-based runner for cross-platform compatibility
Usage: python run.py [action]
Actions: all, install, experiment, evaluate, frontend, notebook, clean
"""
import sys
import os
import subprocess
from pathlib import Path


def print_header(message):
    print("\n" + "="*60)
    print(f"  {message}")
    print("="*60 + "\n")


def run_command(cmd, shell=False):
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, shell=shell, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False


def install_dependencies():
    print_header("Installing Dependencies")
    if run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]):
        print("✅ Dependencies installed successfully")
        return True
    else:
        print("❌ Failed to install dependencies")
        return False


def run_experiments():
    print_header("Running Ablation Experiments")
    print("This may take 5-10 minutes...\n")
    
    if run_command([sys.executable, "run_experiment.py"]):
        print("\n✅ Experiments completed successfully")
        print("Results saved to: experiments/")
        print("Models saved to: artifacts/")
        return True
    else:
        print("❌ Experiments failed")
        return False


def run_evaluation():
    print_header("Running Evaluation and Analysis")
    
    if run_command([sys.executable, "evaluate.py"]):
        print("\n✅ Evaluation completed successfully")
        print("Plots saved to: experiments/plots/")
        print("Report saved to: experiments/analysis_report.txt")
        return True
    else:
        print("❌ Evaluation failed")
        return False


def launch_frontend():
    print_header("Launching Streamlit Frontend")
    print("Opening web interface at http://localhost:8501\n")
    print("Press Ctrl+C to stop the server\n")
    
    os.chdir("frontend")
    run_command([sys.executable, "-m", "streamlit", "run", "app.py"])
    os.chdir("..")


def launch_notebook():
    print_header("Launching Jupyter Notebook")
    print("Opening test_pipeline.ipynb\n")
    run_command([sys.executable, "-m", "jupyter", "notebook", "test_pipeline.ipynb"])


def clean_artifacts():
    print_header("Cleaning Artifacts")
    confirm = input("This will delete all models and results. Continue? (y/N): ")
    
    if confirm.lower() == 'y':
        import shutil
        try:
            if Path("artifacts").exists():
                shutil.rmtree("artifacts")
                Path("artifacts").mkdir()
            if Path("experiments").exists():
                shutil.rmtree("experiments")
                Path("experiments").mkdir()
            print("✅ Artifacts cleaned")
        except Exception as e:
            print(f"❌ Error cleaning artifacts: {e}")
    else:
        print("❌ Cancelled")


def main():
    print("""
╔═══════════════════════════════════════════════════════╗
║                                                       ║
║   🎓 EXAM SCORE PREDICTION PIPELINE                  ║
║                                                       ║
║   Production-Ready ML System                         ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝
    """)
    
    action = sys.argv[1] if len(sys.argv) > 1 else 'all'
    
    if action == 'install':
        install_dependencies()
    
    elif action == 'experiment':
        run_experiments()
    
    elif action == 'evaluate':
        if not Path("experiments/ablation_results.json").exists():
            print("❌ No experiment results found. Run experiments first.")
            sys.exit(1)
        run_evaluation()
    
    elif action == 'frontend':
        if not Path("artifacts/baseline_model.joblib").exists():
            print("⚠️  Warning: No model found. Run experiments first.")
            print("The frontend will use a simplified prediction formula.\n")
        launch_frontend()
    
    elif action == 'notebook':
        launch_notebook()
    
    elif action == 'clean':
        clean_artifacts()
    
    elif action == 'all':
        # Check if dependencies installed
        try:
            import pandas
            print("✅ Dependencies already installed\n")
        except ImportError:
            print("Dependencies not found. Installing...\n")
            if not install_dependencies():
                sys.exit(1)
        
        # Run full workflow
        if not run_experiments():
            sys.exit(1)
        
        if not run_evaluation():
            sys.exit(1)
        
        print_header("Workflow Complete!")
        print("Next steps:")
        print("  1. View results: cat experiments/summary_statistics.txt")
        print("  2. Launch frontend: python run.py frontend")
        print("  3. Explore notebook: python run.py notebook\n")
    
    else:
        print(f"Unknown action: {action}")
        print("Available actions: all, install, experiment, evaluate, frontend, notebook, clean")
        sys.exit(1)
    
    print("\n✨ Done!\n")


if __name__ == '__main__':
    main()
