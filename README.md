# Academic Performance Prediction Framework

A comprehensive machine learning system that predicts student exam scores based on lifestyle factors, sleep quality, caffeine intake, and cognitive states. The system uses a two-stage predictive framework with physiological formulas and ML models to provide actionable insights for academic success.

## 🚀 Features

- **Real-time Score Prediction**: Interactive Streamlit app for predicting exam scores based on daily habits
- **Score Optimization**: AI-powered recommendations for achieving target grades
- **Physiological Modeling**: Evidence-based formulas for sleep quality, mental health, stress, and focus
- **Caffeine Impact Analysis**: Detailed modeling of caffeine's effects on sleep and cognitive performance
- **Comprehensive ML Pipeline**: From data preparation to model deployment with proper validation
- **Docker Deployment**: Containerized application for easy deployment

## 📊 Key Metrics

- **Model Performance**: R² = 0.972 on test set
- **Cross-Validation**: 5-fold CV with R² = 0.956 ± 0.005
- **Dataset**: 10,000 synthetic student records (expanded from 1,000 real records)
- **Features**: 10 engineered features including physiological proxies

## 🛠️ Quick Start

### Docker (Recommended)

```bash
cd production
docker-compose up --build
```

Access the app at `http://localhost:8501`

### Local Setup

1. **Install dependencies**
   ```bash
   pip install -r production/requirements.txt
   ```

2. **Run the application**
   ```bash
   cd production
   streamlit run frontend/app.py
   ```

## 📁 Project Structure

```
CP2025/Project/
├── production/                 # Main application code
│   ├── frontend/              # Streamlit web application
│   │   ├── app.py            # Main Streamlit app
│   │   └── components/       # UI components
│   ├── src/                   # Source code
│   │   ├── pipeline.py       # ML pipeline implementation
│   │   ├── ablation_study.py # Feature importance analysis
│   │   └── model_analysis.py # Model evaluation utilities
│   ├── artifacts/            # Trained models and scalers
│   │   ├── model_randomforest_tuned_model.joblib
│   │   ├── scaler_minmax_model.joblib
│   │   └── encoder_ordinal_model.joblib
│   ├── experiments/          # Experiment results
│   │   ├── ablation_results.json
│   │   └── summary_statistics.txt
│   └── docker-compose.yml    # Docker configuration
└── README.md                 # This file
```

## 🎯 Usage

### Web Application

1. **Access the Streamlit app** at `http://localhost:8501`
2. **Input your daily habits**:
   - Study hours and attendance
   - Sleep duration and quality
   - Caffeine intake (via drink selection)
   - Social media and Netflix usage
   - Exercise frequency
3. **Get predictions** for exam scores and optimization recommendations

### API Usage

The application can be extended to provide REST API endpoints for programmatic access to predictions.

## 🔬 Methodology

### Data Preparation
- **Original Dataset**: 1,000 student records with academic performance data
- **Data Expansion**: Bootstrap sampling with Gaussian noise (5% std deviation) and outliers (8% of samples)
- **Final Dataset**: 10,000 records for robust model training

### Feature Engineering
- **Caffeine Intake**: Calculated from drink selection with demographic distributions
- **Sleep Quality**: Physiological formula incorporating sleep duration, exercise, mental health, and caffeine effects
- **Mental Health**: Lifestyle-based calculation from social media, exercise, and sleep patterns
- **Stress Proxy**: Composite metric combining caffeine, social factors, and sleep quality
- **Focus Proxy**: Cognitive performance indicator based on sleep, caffeine, and behavioral factors

### Model Development
- **Algorithm**: Random Forest with hyperparameter tuning
- **Preprocessing**: Standard scaling for numeric features, one-hot encoding for categorical
- **Validation**: 5-fold cross-validation with grid search optimization
- **Evaluation**: R², MSE, MAE metrics with focus on generalization performance

## 📈 Results

### Model Performance
- **Test Set R²**: 0.972 (excellent predictive power)
- **Cross-Validation R²**: 0.956 ± 0.005 (stable performance)
- **MSE**: 9.02 (low prediction error)

### Feature Importance
Key factors influencing academic performance:
1. Study hours and attendance (direct academic factors)
2. Sleep quality and duration (physiological foundation)
3. Caffeine intake (cognitive modulation)
4. Social media usage (distraction factor)
5. Exercise frequency (mental health indicator)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Research Sources**:
  - Doherty & Smith (2005) - Caffeine pharmacology
  - Cohen et al. (1983) - Perceived stress scale
  - Ruxton (2008) - Caffeine and cognitive performance
  - Kahneman (1973) - Attention theory

- **Dataset**: Original student performance data with synthetic expansion
- **Libraries**: scikit-learn, Streamlit, pandas, numpy, matplotlib

## 📞 Contact

For questions or collaboration opportunities, please reach out to the project maintainers.

---

**Note**: This system is designed for educational and research purposes. Predictions should be used as guidance rather than definitive outcomes, as academic success depends on many individual factors beyond the modeled variables.
