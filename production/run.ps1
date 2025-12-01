# Exam Score Prediction Pipeline - Main Controller
# Run this script to execute the full workflow

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet('all', 'install', 'experiment', 'evaluate', 'frontend', 'notebook', 'clean')]
    [string]$Action = 'all'
)

function Write-Header {
    param([string]$Message)
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  $Message" -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
}

function Install-Dependencies {
    Write-Header "Installing Dependencies"
    pip install -r requirements.txt
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
}

function Run-Experiments {
    Write-Header "Running Ablation Experiments"
    Write-Host "This may take 5-10 minutes...`n" -ForegroundColor Yellow
    python run_experiment.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ Experiments completed successfully" -ForegroundColor Green
        Write-Host "Results saved to: experiments/" -ForegroundColor Green
        Write-Host "Models saved to: artifacts/" -ForegroundColor Green
    } else {
        Write-Host "❌ Experiments failed" -ForegroundColor Red
        exit 1
    }
}

function Run-Evaluation {
    Write-Header "Running Evaluation and Analysis"
    python evaluate.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ Evaluation completed successfully" -ForegroundColor Green
        Write-Host "Plots saved to: experiments/plots/" -ForegroundColor Green
        Write-Host "Report saved to: experiments/analysis_report.txt" -ForegroundColor Green
    } else {
        Write-Host "❌ Evaluation failed" -ForegroundColor Red
        exit 1
    }
}

function Launch-Frontend {
    Write-Header "Launching Streamlit Frontend"
    Write-Host "Opening web interface at http://localhost:8501`n" -ForegroundColor Yellow
    Write-Host "Press Ctrl+C to stop the server`n" -ForegroundColor Yellow
    Set-Location frontend
    streamlit run app.py
    Set-Location ..
}

function Launch-Notebook {
    Write-Header "Launching Jupyter Notebook"
    Write-Host "Opening test_pipeline.ipynb`n" -ForegroundColor Yellow
    jupyter notebook test_pipeline.ipynb
}

function Clean-Artifacts {
    Write-Header "Cleaning Artifacts"
    $confirm = Read-Host "This will delete all models and results. Continue? (y/N)"
    if ($confirm -eq 'y' -or $confirm -eq 'Y') {
        Remove-Item -Path "artifacts\*" -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item -Path "experiments\*" -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "✅ Artifacts cleaned" -ForegroundColor Green
    } else {
        Write-Host "❌ Cancelled" -ForegroundColor Yellow
    }
}

# Main execution
Write-Host @"
╔═══════════════════════════════════════════════════════╗
║                                                       ║
║   🎓 EXAM SCORE PREDICTION PIPELINE                  ║
║                                                       ║
║   Production-Ready ML System                         ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

switch ($Action) {
    'install' {
        Install-Dependencies
    }
    'experiment' {
        Run-Experiments
    }
    'evaluate' {
        if (-not (Test-Path "experiments\ablation_results.json")) {
            Write-Host "❌ No experiment results found. Run experiments first." -ForegroundColor Red
            exit 1
        }
        Run-Evaluation
    }
    'frontend' {
        if (-not (Test-Path "artifacts\baseline_model.joblib")) {
            Write-Host "⚠️  Warning: No model found. Run experiments first." -ForegroundColor Yellow
            Write-Host "The frontend will use a simplified prediction formula.`n" -ForegroundColor Yellow
        }
        Launch-Frontend
    }
    'notebook' {
        Launch-Notebook
    }
    'clean' {
        Clean-Artifacts
    }
    'all' {
        # Check if dependencies installed
        $checkPandas = python -c "import pandas; print('ok')" 2>$null
        if ($checkPandas -ne 'ok') {
            Write-Host "Dependencies not found. Installing...`n" -ForegroundColor Yellow
            Install-Dependencies
        } else {
            Write-Host "✅ Dependencies already installed`n" -ForegroundColor Green
        }
        
        # Run full workflow
        Run-Experiments
        Run-Evaluation
        
        Write-Header "Workflow Complete!"
        Write-Host "Next steps:" -ForegroundColor Green
        Write-Host "  1. View results: cat experiments\summary_statistics.txt" -ForegroundColor White
        Write-Host "  2. Launch frontend: .\run.ps1 frontend" -ForegroundColor White
        Write-Host "  3. Explore notebook: .\run.ps1 notebook`n" -ForegroundColor White
    }
}

Write-Host "`n✨ Done!`n" -ForegroundColor Green
