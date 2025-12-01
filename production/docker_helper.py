#!/usr/bin/env python
"""
Docker Helper Script - Cross-platform
Usage: python docker_helper.py [action]
"""
import sys
import subprocess
import os


def run_cmd(cmd):
    """Run a shell command."""
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def show_help():
    print("""
╔═══════════════════════════════════════════════════════╗
║   Docker Helper for Exam Score Predictor             ║
╚═══════════════════════════════════════════════════════╝

Usage: python docker_helper.py [action]

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
  python docker_helper.py build
  python docker_helper.py up
  python docker_helper.py experiments
  python docker_helper.py logs
  python docker_helper.py shell
    """)


def build_images():
    print("🔨 Building Docker images...")
    if run_cmd("docker-compose build"):
        print("✅ Build completed successfully")
    else:
        print("❌ Build failed")


def start_services():
    print("🚀 Starting frontend service...")
    if run_cmd("docker-compose up -d frontend"):
        print("✅ Frontend started successfully")
        print("🌐 Access at: http://localhost:8501")
    else:
        print("❌ Failed to start services")


def stop_services():
    print("🛑 Stopping services...")
    run_cmd("docker-compose down")
    print("✅ Services stopped")


def show_logs():
    print("📋 Showing logs (Ctrl+C to exit)...")
    run_cmd("docker-compose logs -f")


def open_shell():
    print("🐚 Opening shell in frontend container...")
    run_cmd("docker exec -it exam-predictor-frontend /bin/bash")


def run_experiments():
    print("🧪 Running experiments in container...")
    print("This may take 5-10 minutes...")
    if run_cmd("docker-compose --profile experiments run --rm experiment-runner"):
        print("✅ Experiments completed")
    else:
        print("❌ Experiments failed")


def run_evaluation():
    print("📊 Running evaluation in container...")
    if run_cmd("docker-compose --profile evaluation run --rm evaluator"):
        print("✅ Evaluation completed")
    else:
        print("❌ Evaluation failed")


def clean_all():
    print("🗑️  Cleaning Docker resources...")
    confirm = input("Remove all containers, volumes, and images? (y/N): ")
    if confirm.lower() == 'y':
        run_cmd("docker-compose down -v --rmi all")
        print("✅ Cleanup completed")
    else:
        print("❌ Cancelled")


def start_production():
    print("🚢 Building and starting production stack...")
    run_cmd("docker-compose -f docker-compose.prod.yml build")
    if run_cmd("docker-compose -f docker-compose.prod.yml up -d"):
        print("✅ Production stack started")
        print("🌐 Access at: http://localhost:8501")
    else:
        print("❌ Failed to start production stack")


def main():
    actions = {
        'build': build_images,
        'up': start_services,
        'down': stop_services,
        'logs': show_logs,
        'shell': open_shell,
        'experiments': run_experiments,
        'evaluate': run_evaluation,
        'clean': clean_all,
        'prod': start_production,
        'help': show_help
    }
    
    action = sys.argv[1] if len(sys.argv) > 1 else 'help'
    
    if action in actions:
        actions[action]()
    else:
        print(f"Unknown action: {action}")
        show_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
