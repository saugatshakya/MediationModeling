"""
Configurable Regression Pipeline for Exam Score Prediction
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from typing import Dict, Any, Tuple, Optional


class ExamScorePipeline:
    """
    Flexible pipeline for exam score prediction with configurable preprocessing and models.
    """
    
    def __init__(
        self,
        scaler_type: str = 'standard',
        encoder_type: str = 'onehot',
        model_type: str = 'randomforest',
        random_state: int = 42,
        model_kwargs: dict = None
    ):
        """
        Initialize the pipeline with specified components.
        
        Args:
            scaler_type: 'standard', 'minmax', 'robust', or 'none'
            encoder_type: 'onehot' or 'ordinal'
            model_type: 'randomforest', 'gradientboosting', 'linear', 'ridge', 'lasso', 'svr', 'knn'
            random_state: Random seed for reproducibility
            model_kwargs: Additional parameters for the model
        """
        self.scaler_type = scaler_type
        self.encoder_type = encoder_type
        self.model_type = model_type
        self.random_state = random_state
        self.model_kwargs = model_kwargs or {}
        self.pipeline = None
        self.feature_names = None
        self.numeric_features = None
        self.categorical_features = None
        
    def _get_scaler(self):
        """Return scaler based on type."""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'none': 'passthrough'
        }
        return scalers.get(self.scaler_type, StandardScaler())
    
    def _get_encoder(self):
        """Return encoder based on type."""
        encoders = {
            'onehot': OneHotEncoder(handle_unknown='ignore', sparse_output=False),
            'ordinal': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        }
        return encoders.get(self.encoder_type, OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    
    def _get_model(self):
        """Return model based on type."""
        try:
            if self.model_type == 'randomforest':
                model = RandomForestRegressor(random_state=self.random_state, **self.model_kwargs)
            elif self.model_type == 'gradientboosting':
                model = GradientBoostingRegressor(random_state=self.random_state, **self.model_kwargs)
            elif self.model_type == 'linear':
                model = LinearRegression(**self.model_kwargs)
            elif self.model_type == 'ridge':
                model = Ridge(random_state=self.random_state, **self.model_kwargs)
            elif self.model_type == 'lasso':
                model = Lasso(random_state=self.random_state, **self.model_kwargs)
            elif self.model_type == 'svr':
                model = SVR(**self.model_kwargs)
            elif self.model_type == 'knn':
                model = KNeighborsRegressor(**self.model_kwargs)
            else:
                model = RandomForestRegressor(random_state=self.random_state)
            return model
        except Exception as e:
            print(f"Error creating model {self.model_type}: {e}")
            raise
    
    def calculate_sleep_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sleep_quality using the formula if it's not already present
        and all required features are available.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with sleep_quality calculated
        """
        df = df.copy()
        
        if 'sleep_quality' not in df.columns:
            # Check if all required features are present
            required_features = ['sleep_hours', 'exercise_frequency', 'caffeine_intake']
            missing_features = [f for f in required_features if f not in df.columns]
            
            if missing_features:
                # Cannot calculate sleep_quality, skip
                print(f"Warning: Cannot calculate sleep_quality, missing features: {missing_features}")
                return df
            
            # Map categorical features to numeric
            diet_map = {'Poor': -1, 'Fair': 0, 'Good': 1}
            internet_map = {'Poor': -1, 'Average': 0, 'Good': 1}
            
            # Get required features
            sleep = df['sleep_hours']
            exercise = df['exercise_frequency']
            mental = df.get('mental_health_rating', 7)  # Default if not present
            caffeine = df['caffeine_intake']
            diet = df.get('diet_quality', 'Fair')  # Default if not present
            internet = df.get('internet_quality', 'Average')  # Default if not present
            
            # Convert categorical to numeric
            diet_score = pd.Series(diet).map(diet_map).fillna(0)
            internet_score = pd.Series(internet).map(internet_map).fillna(0)
            
            # Sleep quality formula (same as in main notebook)
            sleep_quality = (
                0.5 * sleep +
                0.3 * exercise +
                0.2 * mental -
                0.3 * caffeine +
                0.5 * diet_score +
                0.3 * internet_score +
                np.random.normal(0, 0.8, len(df))
            )
            
            # Scale to 1-10
            df['sleep_quality'] = np.clip(sleep_quality, 1, 10).round(1)
        
        return df
    
    def build_pipeline(self, numeric_features: list, categorical_features: list):
        """
        Build the complete pipeline with preprocessing and model.
        
        Args:
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names
        """
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        
        # Numeric preprocessing
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', self._get_scaler())
        ])
        
        # Categorical preprocessing
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', self._get_encoder())
        ])
        
        # Combine preprocessors
        transformers = [('num', numeric_transformer, numeric_features)]
        
        if categorical_features:
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        # Build full pipeline
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', self._get_model())
        ])
        
        return self.pipeline
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit the pipeline on training data."""
        if self.pipeline is None:
            raise ValueError("Pipeline not built. Call build_pipeline() first.")
        
        # Calculate sleep_quality if needed
        X = self.calculate_sleep_quality(X)
        
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X: pd.DataFrame):
        """Make predictions on new data."""
        if self.pipeline is None:
            raise ValueError("Pipeline not built or fitted.")
        
        # Calculate sleep_quality if needed
        X = self.calculate_sleep_quality(X)
        
        return self.pipeline.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the pipeline on test data with comprehensive metrics.
        
        Returns:
            Dictionary with various performance metrics
        """
        y_pred = self.predict(X)
        
        # Standard metrics
        mse = mean_squared_error(y, y_pred)
        metrics = {
            'mse': mse,
            'r2': r2_score(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mse)
        }
        
        # Custom loss functions
        # Mean Absolute Percentage Error
        metrics['mape'] = np.mean(np.abs((y - y_pred) / np.maximum(np.abs(y), 1e-10))) * 100
        
        # Symmetric Mean Absolute Percentage Error
        metrics['smape'] = 100 * np.mean(2 * np.abs(y - y_pred) / (np.abs(y) + np.abs(y_pred) + 1e-10))
        
        # Mean Absolute Scaled Error (compared to naive forecast)
        naive_pred = np.roll(y, 1)
        naive_pred[0] = y.mean()  # First value uses mean
        naive_mae = mean_absolute_error(y, naive_pred)
        metrics['mase'] = mean_absolute_error(y, y_pred) / naive_mae if naive_mae > 0 else np.inf
        
        # Root Mean Square Percentage Error
        metrics['rmspe'] = np.sqrt(np.mean(((y - y_pred) / np.maximum(np.abs(y), 1e-10))**2)) * 100
        
        # Huber loss (delta=1.0)
        delta = 1.0
        residuals = y - y_pred
        metrics['huber_loss'] = np.sum(
            np.where(np.abs(residuals) <= delta, 
                    0.5 * residuals**2, 
                    delta * (np.abs(residuals) - 0.5 * delta))
        ) / len(y)
        
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Returns:
            Dictionary with mean and std of scores
        """
        scores = cross_val_score(
            self.pipeline, X, y, 
            cv=cv, 
            scoring='r2',
            n_jobs=-1
        )
        
        return {
            'cv_r2_mean': scores.mean(),
            'cv_r2_std': scores.std(),
            'cv_scores': scores
        }
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance for tree-based and linear models.
        
        Returns:
            DataFrame with feature names and importance scores, or None if not applicable
        """
        model = self.pipeline.named_steps['regressor']
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            preprocessor = self.pipeline.named_steps['preprocessor']
            
            # Get feature names from transformers
            feature_names = []
            
            # Numeric features
            feature_names.extend(self.numeric_features)
            
            # Categorical features (after encoding) - only if they exist
            if self.categorical_features:
                if self.encoder_type == 'onehot':
                    try:
                        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
                        if hasattr(cat_encoder, 'get_feature_names_out'):
                            cat_features = cat_encoder.get_feature_names_out(self.categorical_features)
                            feature_names.extend(cat_features)
                        else:
                            feature_names.extend(self.categorical_features)
                    except:
                        # Encoder not fitted or no categorical features
                        pass
                elif self.encoder_type == 'ordinal':
                    feature_names.extend(self.categorical_features)
            
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(model.feature_importances_)],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        elif hasattr(model, 'coef_'):
            # Linear models - use absolute coefficients as importance
            preprocessor = self.pipeline.named_steps['preprocessor']
            
            # Get feature names from transformers
            feature_names = []
            
            # Numeric features
            feature_names.extend(self.numeric_features)
            
            # Categorical features (after encoding)
            if self.encoder_type == 'onehot' and self.categorical_features:
                try:
                    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
                    if hasattr(cat_encoder, 'get_feature_names_out'):
                        cat_features = cat_encoder.get_feature_names_out(self.categorical_features)
                        feature_names.extend(cat_features)
                    else:
                        feature_names.extend(self.categorical_features)
                except:
                    # Encoder not fitted or no categorical features
                    pass
            elif self.encoder_type == 'ordinal' and self.categorical_features:
                feature_names.extend(self.categorical_features)
            
            # Handle multi-output case (though we have single output)
            coef = model.coef_
            if coef.ndim > 1:
                coef = coef[0]
            
            # Ensure we don't exceed available coefficients
            n_features = min(len(feature_names), len(coef))
            importance_df = pd.DataFrame({
                'feature': feature_names[:n_features],
                'importance': np.abs(coef[:n_features])
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return None
    
    def save(self, filepath: str):
        """Save the pipeline to disk."""
        joblib.dump(self.pipeline, filepath)
        
    @classmethod
    def load(cls, filepath: str):
        """Load a saved pipeline."""
        instance = cls()
        instance.pipeline = joblib.load(filepath)
        return instance


def prepare_features(df: pd.DataFrame) -> Tuple[list, list]:
    """
    Identify numeric and categorical features from dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        Tuple of (numeric_features, categorical_features)
    """
    # Exclude target, ID columns, and features dropped in main.ipynb
    exclude_cols = [
        'exam_score', 'student_id', 'gender', 'diet_score', 
        'extracurricular_participation', 'age', 'part_job_score', 
        'part_time_job', 'internet_quality', 'internet_score', 
        'parental_education_level', 'diet_quality', 'mental_health_rating'
    ]
    
    # Identify feature types
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove excluded columns
    numeric_features = [col for col in numeric_features if col not in exclude_cols]
    categorical_features = [col for col in categorical_features if col not in exclude_cols]
    
    return numeric_features, categorical_features
