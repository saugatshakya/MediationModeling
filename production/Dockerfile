# Multi-stage Dockerfile for Exam Score Prediction Pipeline

# Stage 1: Base Python environment
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Application
FROM base as app

# Copy application code
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY run_experiment.py .
COPY evaluate.py .
COPY run.py .

# Create directories for artifacts and experiments
RUN mkdir -p artifacts experiments

# Copy data (if included in image, otherwise mount as volume)
# COPY ../data ./data

# Expose Streamlit default port
EXPOSE 8501

# Default command runs the frontend
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
