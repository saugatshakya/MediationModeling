"""
Ablation Study Framework for Feature Ablation and Hyperparameter Tuning
"""
import numpy as np
import pandas as pd
from itertools import product
from typing import Dict, List, Any
import json
import time
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from pipeline import ExamScorePipeline, prepare_features
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class AblationStudy:
    """
    Framework for running systematic ablation studies on the regression pipeline.
    """
    
    def __init__(self, data_path: str, random_state: int = 42):
        """
        Initialize ablation study.
        
        Args:
            data_path: Path to the processed dataset
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path
        self.random_state = random_state
        self.results = []
        
        # Set random seeds
        np.random.seed(random_state)
        
    def define_experiments(self) -> List[Dict[str, Any]]:
        """
        Define all ablation experiments to run.
        Focus on feature ablation: systematically removing feature groups.
        
        Returns:
            List of experiment configurations
        """
        # Feature groups for ablation
        all_features = [
            'study_hours_per_day', 'social_media_hours', 'netflix_hours', 
            'attendance_percentage', 'sleep_hours', 'exercise_frequency',
            'caffeine_intake', 'sleep_quality', 'stress_proxy', 'focus_proxy'
        ]
        
        feature_groups = {
            'study_habits': ['study_hours_per_day', 'attendance_percentage'],
            'screen_time': ['social_media_hours', 'netflix_hours'],
            'lifestyle': ['sleep_hours', 'exercise_frequency', 'caffeine_intake', 'sleep_quality'],
            'synthetic': ['stress_proxy', 'focus_proxy']
        }
        
        experiments = []
        
        # Baseline: all features
        baseline = {
            'name': 'baseline_all_features',
            'features_to_use': all_features,
            'scaler_type': 'standard',
            'encoder_type': 'onehot',
            'model_type': 'linear',
            'model_kwargs': {},
            'do_hyperparameter_tuning': False
        }
        experiments.append(baseline)
        
        # Ablation 1: Remove each feature group
        for group_name, features in feature_groups.items():
            exp = baseline.copy()
            exp['name'] = f'ablate_{group_name}'
            exp['features_to_use'] = [f for f in all_features if f not in features]
            experiments.append(exp)
        
        # Ablation 2: Keep only one group at a time
        for group_name, features in feature_groups.items():
            exp = baseline.copy()
            exp['name'] = f'only_{group_name}'
            exp['features_to_use'] = features
            experiments.append(exp)
        
        # Ablation 3: Model variations with hyperparameter tuning
        model_configs = [
            {
                'model_type': 'ridge',
                'param_grid': {'alpha': [0.1, 1.0, 10.0]},
                'do_hyperparameter_tuning': True
            },
            {
                'model_type': 'lasso', 
                'param_grid': {'alpha': [0.01, 0.1, 1.0]},
                'do_hyperparameter_tuning': True
            },
            {
                'model_type': 'randomforest',
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None]
                },
                'do_hyperparameter_tuning': True
            },
            {
                'model_type': 'gradientboosting',
                'param_grid': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2]
                },
                'do_hyperparameter_tuning': True
            }
        ]
        
        for config in model_configs:
            exp = baseline.copy()
            exp['name'] = f'model_{config["model_type"]}_tuned'
            exp['model_type'] = config['model_type']
            exp['param_grid'] = config['param_grid']
            exp['do_hyperparameter_tuning'] = config['do_hyperparameter_tuning']
            experiments.append(exp)
        
        return experiments
    
    def run_single_experiment(
        self,
        config: Dict[str, Any],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        all_numeric_features: List[str],
        all_categorical_features: List[str]
    ) -> Dict[str, Any]:
        """
        Run a single experiment with given configuration.
        
        Args:
            config: Experiment configuration
            X_train, X_test, y_train, y_test: Train/test splits
            all_numeric_features: All available numeric feature names
            all_categorical_features: All available categorical feature names
            
        Returns:
            Dictionary with experiment results
        """
        print(f"\nRunning experiment: {config['name']}")
        print(f"  Features: {len(config['features_to_use'])} features")
        print(f"  Scaler: {config['scaler_type']}")
        print(f"  Encoder: {config['encoder_type']}")
        print(f"  Model: {config['model_type']}")
        print(f"  Hyperparameter tuning: {config.get('do_hyperparameter_tuning', False)}")
        
        # Feature selection
        selected_features = config['features_to_use']
        numeric_features = [f for f in selected_features if f in all_numeric_features]
        categorical_features = [f for f in selected_features if f in all_categorical_features]
        
        # Filter datasets to selected features
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        print(f"  Selected features: {selected_features}")
        
        # Initialize pipeline
        pipeline = ExamScorePipeline(
            scaler_type=config['scaler_type'],
            encoder_type=config['encoder_type'],
            model_type=config['model_type'],
            random_state=self.random_state,
            model_kwargs=config.get('model_kwargs', {})
        )
        
        # Build pipeline
        pipeline.build_pipeline(numeric_features, categorical_features)
        
        # Hyperparameter tuning with CV
        best_params = None
        cv_scores = None
        if config.get('do_hyperparameter_tuning', False) and 'param_grid' in config:
            print(f"  Performing hyperparameter tuning...")
            
            # Create parameter grid for the model
            param_grid = {f'regressor__{k}': v for k, v in config['param_grid'].items()}
            
            # Custom scorers
            scorers = {
                'mse': make_scorer(mean_squared_error, greater_is_better=False),
                'mae': make_scorer(mean_absolute_error, greater_is_better=False),
                'r2': make_scorer(r2_score),
                'rmse': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)
            }
            
            # Grid search with CV
            grid_search = GridSearchCV(
                pipeline.pipeline,
                param_grid,
                cv=5,
                scoring=scorers,
                refit='r2',
                n_jobs=-1,
                verbose=0
            )
            
            start_time = time.time()
            grid_search.fit(X_train_selected, y_train)
            tuning_time = time.time() - start_time
            
            # Update pipeline with best model
            pipeline.pipeline = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_scores = {
                'best_score': grid_search.best_score_,
                'mse_scores': -grid_search.cv_results_[f'mean_test_mse'],
                'mae_scores': -grid_search.cv_results_[f'mean_test_mae'],
                'r2_scores': grid_search.cv_results_[f'mean_test_r2'],
                'rmse_scores': -grid_search.cv_results_[f'mean_test_rmse']
            }
            
            print(f"  Best params: {best_params}")
            print(f"  CV R²: {grid_search.best_score_:.4f} (+/- {grid_search.cv_results_['std_test_r2'][grid_search.best_index_]:.4f})")
        
        # Time training
        start_time = time.time()
        pipeline.fit(X_train_selected, y_train)
        training_time = time.time() - start_time
        
        # Evaluate on test set
        test_metrics = pipeline.evaluate(X_test_selected, y_test)
        
        # Cross-validation on full training set
        if cv_scores is None:
            cv_scores = cross_val_score(pipeline.pipeline, X_train_selected, y_train, cv=5, scoring='r2')
            cv_scores = {'r2_scores': cv_scores, 'mean_cv_r2': cv_scores.mean(), 'std_cv_r2': cv_scores.std()}
        
        # Time inference
        start_time = time.time()
        _ = pipeline.predict(X_test_selected)
        inference_time = time.time() - start_time
        
        # Get feature importance if available
        feature_importance = None
        if config['model_type'] in ['randomforest', 'gradientboosting', 'ridge', 'lasso', 'linear']:
            importance_df = pipeline.get_feature_importance()
            if importance_df is not None:
                feature_importance = importance_df.head(10).to_dict('records')
        
        # Compile results
        result = {
            'experiment_name': config['name'],
            'config': config,
            'selected_features': selected_features,
            'metrics': test_metrics,
            'cv_scores': cv_scores,
            'training_time': training_time,
            'inference_time': inference_time,
            'best_params': best_params,
            'feature_importance': feature_importance
        }
        
        print(f"  Test Results: MSE={test_metrics['mse']:.2f}, R²={test_metrics['r2']:.4f}")
        if 'mean_cv_r2' in cv_scores:
            print(f"  CV Results: R²={cv_scores['mean_cv_r2']:.4f} (+/- {cv_scores['std_cv_r2']:.4f})")
        
        return result
    
    def run_all_experiments(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        save_models: bool = True,
        artifacts_dir: str = '../artifacts'
    ):
        """
        Run all defined ablation experiments.
        
        Args:
            df: Full dataset
            test_size: Proportion for test split
            save_models: Whether to save trained models
            artifacts_dir: Directory to save models
        """
        # Prepare features
        numeric_features, categorical_features = prepare_features(df)
        
        # Prepare data - drop columns as per main.ipynb notebook
        # These columns were excluded in the final model training
        columns_to_drop = [
            'exam_score', 'student_id', 'gender', 'diet_score', 
            'extracurricular_participation', 'age', 'part_job_score', 
            'part_time_job', 'internet_quality', 'internet_score', 
            'parental_education_level', 'diet_quality', 'mental_health_rating'
        ]
        X = df.drop(columns=columns_to_drop, errors='ignore')
        y = df['exam_score']
        
        # Split data (fixed split for fair comparison)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"Dataset split: {len(X_train)} train, {len(X_test)} test")
        print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
        print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
        
        # Define and run experiments
        experiments = self.define_experiments()
        print(f"\nTotal experiments to run: {len(experiments)}")
        
        self.results = []
        
        for config in experiments:
            try:
                result = self.run_single_experiment(
                    config, X_train, X_test, y_train, y_test,
                    numeric_features, categorical_features
                )
                self.results.append(result)
                
                # Save model if requested
                if save_models:
                    artifacts_path = Path(artifacts_dir)
                    artifacts_path.mkdir(parents=True, exist_ok=True)
                    
                    # Rebuild pipeline to save
                    pipeline = ExamScorePipeline(
                        scaler_type=config['scaler_type'],
                        encoder_type=config['encoder_type'],
                        model_type=config['model_type'],
                        random_state=self.random_state,
                        model_kwargs=config.get('model_kwargs', {})
                    )
                    pipeline.build_pipeline(numeric_features, categorical_features)
                    pipeline.fit(X_train, y_train)
                    
                    model_path = artifacts_path / f"{config['name']}_model.joblib"
                    pipeline.save(str(model_path))
                    
            except Exception as e:
                print(f"  ERROR in {config['name']}: {str(e)}")
                continue
        
        print("\n" + "="*50)
        print("All experiments completed!")
        print(f"Successful: {len(self.results)}/{len(experiments)}")
        
        return self.results
    
    def save_results(self, output_dir: str = '../experiments'):
        """
        Save experiment results to JSON and CSV.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save full results as JSON
        json_path = output_path / 'ablation_results.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Create summary DataFrame
        summary_data = []
        for result in self.results:
            summary_data.append({
                'experiment': result['experiment_name'],
                'scaler': result['config']['scaler_type'],
                'encoder': result['config']['encoder_type'],
                'model': result['config']['model_type'],
                'mse': result['metrics']['mse'],
                'r2': result['metrics']['r2'],
                'mae': result['metrics']['mae'],
                'rmse': result['metrics']['rmse'],
                'training_time': result['training_time'],
                'inference_time': result['inference_time']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('r2', ascending=False)
        
        # Save as CSV
        csv_path = output_path / 'ablation_summary.csv'
        summary_df.to_csv(csv_path, index=False)
        
        print(f"\nResults saved to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")
        
        return summary_df
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary of all experiments as DataFrame."""
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        for result in self.results:
            summary_data.append({
                'experiment': result['experiment_name'],
                'scaler': result['config']['scaler_type'],
                'encoder': result['config']['encoder_type'],
                'model': result['config']['model_type'],
                'mse': result['metrics']['mse'],
                'r2': result['metrics']['r2'],
                'mae': result['metrics']['mae'],
                'rmse': result['metrics']['rmse'],
                'training_time': result['training_time']
            })
        
        return pd.DataFrame(summary_data).sort_values('r2', ascending=False)
