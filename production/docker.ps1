# Docker Build and Run Helper Script for Windows PowerShell

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet('build', 'up', 'down', 'logs', 'shell', 'experiments', 'evaluate', 'clean', 'prod', 'help')]
    [string]$Action = 'help'
)

function Show-Help {
    Write-Host @"
╔═══════════════════════════════════════════════════════╗
║   Docker Helper for Exam Score Predictor             ║
╚═══════════════════════════════════════════════════════╝

Usage: .\docker.ps1 [action]

Actions:
  build       Build Docker images
  up          Start frontend service
  down        Stop all services
  logs        View logs (follow mode)
  shell       Open shell in frontend container
  experiments Run experiments in container
  evaluate    Run evaluation in container
  clean       Remove containers, volumes, images
  prod        Build and start production stack
  help        Show this help message

Examples:
  .\docker.ps1 build
  .\docker.ps1 up
  .\docker.ps1 experiments
  .\docker.ps1 logs
  .\docker.ps1 shell

"@ -ForegroundColor Cyan
}

function Build-Images {
    Write-Host "🔨 Building Docker images..." -ForegroundColor Cyan
    docker-compose build
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Build completed successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Build failed" -ForegroundColor Red
    }
}

function Start-Services {
    Write-Host "🚀 Starting frontend service..." -ForegroundColor Cyan
    docker-compose up -d frontend
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Frontend started successfully" -ForegroundColor Green
        Write-Host "🌐 Access at: http://localhost:8501" -ForegroundColor Yellow
    } else {
        Write-Host "❌ Failed to start services" -ForegroundColor Red
    }
}

function Stop-Services {
    Write-Host "🛑 Stopping services..." -ForegroundColor Cyan
    docker-compose down
    Write-Host "✅ Services stopped" -ForegroundColor Green
}

function Show-Logs {
    Write-Host "📋 Showing logs (Ctrl+C to exit)..." -ForegroundColor Cyan
    docker-compose logs -f
}

function Open-Shell {
    Write-Host "🐚 Opening shell in frontend container..." -ForegroundColor Cyan
    docker exec -it exam-predictor-frontend /bin/bash
}

function Run-Experiments {
    Write-Host "🧪 Running experiments in container..." -ForegroundColor Cyan
    Write-Host "This may take 5-10 minutes..." -ForegroundColor Yellow
    docker-compose --profile experiments run --rm experiment-runner
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Experiments completed" -ForegroundColor Green
    } else {
        Write-Host "❌ Experiments failed" -ForegroundColor Red
    }
}

function Run-Evaluation {
    Write-Host "📊 Running evaluation in container..." -ForegroundColor Cyan
    docker-compose --profile evaluation run --rm evaluator
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Evaluation completed" -ForegroundColor Green
    } else {
        Write-Host "❌ Evaluation failed" -ForegroundColor Red
    }
}

function Clean-All {
    Write-Host "🗑️  Cleaning Docker resources..." -ForegroundColor Yellow
    $confirm = Read-Host "Remove all containers, volumes, and images? (y/N)"
    if ($confirm -eq 'y' -or $confirm -eq 'Y') {
        docker-compose down -v --rmi all
        Write-Host "✅ Cleanup completed" -ForegroundColor Green
    } else {
        Write-Host "❌ Cancelled" -ForegroundColor Yellow
    }
}

function Start-Production {
    Write-Host "🚢 Building and starting production stack..." -ForegroundColor Cyan
    docker-compose -f docker-compose.prod.yml build
    docker-compose -f docker-compose.prod.yml up -d
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Production stack started" -ForegroundColor Green
        Write-Host "🌐 Access at: http://localhost:8501" -ForegroundColor Yellow
    } else {
        Write-Host "❌ Failed to start production stack" -ForegroundColor Red
    }
}

# Main execution
switch ($Action) {
    'build'       { Build-Images }
    'up'          { Start-Services }
    'down'        { Stop-Services }
    'logs'        { Show-Logs }
    'shell'       { Open-Shell }
    'experiments' { Run-Experiments }
    'evaluate'    { Run-Evaluation }
    'clean'       { Clean-All }
    'prod'        { Start-Production }
    'help'        { Show-Help }
}
