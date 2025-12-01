"""
Main Experiment Runner Script

This script loads the data, runs ablation studies, and saves results.
Usage: python run_experiment.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from ablation_study import AblationStudy


def load_and_prepare_data(data_dir: str = '../data') -> pd.DataFrame:
    """
    Load the student habits dataset and generate synthetic features.
    
    Args:
        data_dir: Directory containing CSV files
        
    Returns:
        Processed DataFrame ready for modeling
    """
    print("Loading data...")
    data_path = Path(data_dir) / 'student_habits_performance.csv'
    df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} rows from {data_path}")
    
    # Generate caffeine and sleep quality features
    df = generate_caffeine_and_sleep(df)
    
    # Expand dataset to 10k rows
    df = expand_dataset(df, target_size=10000)
    
    # Generate stress and focus proxies
    df = generate_focus_and_stress(df)
    
    print(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
    
    return df


def generate_caffeine_and_sleep(df: pd.DataFrame) -> pd.DataFrame:
    """Generate caffeine intake feature (sleep quality calculated by pipeline)."""
    df = df.copy()

    # Map categorical features to numeric influences
    diet_map = {'Poor': -1, 'Fair': 0, 'Good': 1}
    internet_map = {'Poor': -1, 'Average': 0, 'Good': 1}

    # Normalize numeric features
    study = df['study_hours_per_day']
    social = df['social_media_hours']
    netflix = df['netflix_hours']
    sleep = df['sleep_hours']
    exercise = df['exercise_frequency']
    mental = df['mental_health_rating']
    part_job = df['part_time_job'].map({'Yes': 1, 'No': 0})
    diet = df['diet_quality'].map(diet_map).fillna(0)
    internet = df['internet_quality'].map(internet_map).fillna(0)

    # Caffeine intake formula
    caffeine = (
        0.4 * study +
        0.3 * social +
        0.2 * netflix +
        0.2 * (10 - sleep) +
        1.5 * part_job +
        np.random.normal(0, 0.8, len(df))
    )

    # Scale caffeine to 1–10 (sleep_quality will be calculated by pipeline)
    df['caffeine_intake'] = np.clip(caffeine, 1, 10).round(1)

    return df


def expand_dataset(df: pd.DataFrame, target_size: int = 10000) -> pd.DataFrame:
    """Expand dataset by sampling with noise."""
    df = df.copy()
    n_current = len(df)
    n_needed = target_size - n_current

    # Sample existing data to create new ones
    new_data = df.sample(n=n_needed, replace=True, random_state=42).reset_index(drop=True)

    # Add small Gaussian noise to numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if col not in ['student_id']:
            noise = np.random.normal(0, df[col].std() * 0.05, n_needed)
            new_data[col] = np.clip(new_data[col] + noise, df[col].min(), df[col].max())

    # Add random outliers (about 8%)
    n_outliers = int(0.08 * n_needed)
    outlier_rows = new_data.sample(n=n_outliers, replace=False, random_state=7)
    for col in ['caffeine_intake', 'exam_score']:
        if col in new_data.columns:
            new_data.loc[outlier_rows.index, col] = np.clip(
                new_data.loc[outlier_rows.index, col] + np.random.normal(0, 3, n_outliers),
                1, 10 if col != 'exam_score' else 100
            )

    # Create unique student IDs only for new_data
    start_id = 1000 + n_current
    new_data['student_id'] = [f"S{start_id + i}" for i in range(n_needed)]

    # Combine old + new
    expanded_df = pd.concat([df, new_data], ignore_index=True)
    expanded_df = expanded_df.reset_index(drop=True)

    return expanded_df


def generate_focus_and_stress(df: pd.DataFrame) -> pd.DataFrame:
    """Generate stress and focus proxy features."""
    df = df.copy()

    # Calculate sleep_quality if not present (using same formula as pipeline)
    if 'sleep_quality' not in df.columns:
        diet_map = {'Poor': -1, 'Fair': 0, 'Good': 1}
        internet_map = {'Poor': -1, 'Average': 0, 'Good': 1}
        
        sleep = df['sleep_hours']
        exercise = df['exercise_frequency']
        mental = df['mental_health_rating']
        caffeine = df['caffeine_intake']
        diet = df['diet_quality'].map(diet_map).fillna(0)
        internet = df['internet_quality'].map(internet_map).fillna(0)
        
        sleep_quality = (
            0.5 * sleep +
            0.3 * exercise +
            0.2 * mental -
            0.3 * caffeine +
            0.5 * diet +
            0.3 * internet +
            np.random.normal(0, 0.8, len(df))
        )
        df['sleep_quality'] = np.clip(sleep_quality, 1, 10).round(1)

    # --- Base normalization ---
    Sd = (df['sleep_hours'] - df['sleep_hours'].min()) / (df['sleep_hours'].max() - df['sleep_hours'].min())
    Sq = (df['sleep_quality'] - df['sleep_quality'].min()) / (df['sleep_quality'].max() - df['sleep_quality'].min())

    # Map caffeine_intake (1–10) → mg/day range [100, 500]
    C_mg = 50 + 50 * df['caffeine_intake']

    # Caffeine functions
    fC = np.exp(-((C_mg - 200) ** 2) / (2 * (100 ** 2)))
    gC = np.maximum(0, (C_mg - 300) / 200)

    # Map categorical features
    diet_map = {'Poor': -1, 'Fair': 0, 'Good': 1}
    internet_map = {'Poor': -1, 'Average': 0, 'Good': 1}
    part_job_map = {'Yes': 1, 'No': 0}
    df['diet_score'] = df['diet_quality'].map(diet_map).fillna(0)
    df['internet_score'] = df['internet_quality'].map(internet_map).fillna(0)
    df['part_job_score'] = df['part_time_job'].map(part_job_map).fillna(0)

    # Normalize continuous inputs
    study = (df['study_hours_per_day'] - df['study_hours_per_day'].min()) / (df['study_hours_per_day'].max() - df['study_hours_per_day'].min())
    social = (df['social_media_hours'] - df['social_media_hours'].min()) / (df['social_media_hours'].max() - df['social_media_hours'].min())
    netflix = (df['netflix_hours'] - df['netflix_hours'].min()) / (df['netflix_hours'].max() - df['netflix_hours'].min())
    attend = (df['attendance_percentage'] - df['attendance_percentage'].min()) / (df['attendance_percentage'].max() - df['attendance_percentage'].min())
    exercise = df['exercise_frequency'] / df['exercise_frequency'].max()
    mental = df['mental_health_rating'] / 10

    # Compute base stress & focus
    Str_p = gC - Sq
    F_p = Sd + Sq - Str_p + fC

    # Add effects of other parameters
    Str_p += (
        + 0.4 * social
        + 0.3 * netflix
        + 0.2 * study
        + 0.3 * df['part_job_score']
        - 0.3 * exercise
        - 0.4 * mental
        - 0.2 * df['diet_score']
        - 0.2 * df['internet_score']
    )

    F_p += (
        + 0.4 * exercise
        + 0.4 * mental
        + 0.3 * attend
        + 0.2 * df['diet_score']
        + 0.2 * df['internet_score']
        - 0.3 * social
        - 0.2 * netflix
        - 0.2 * df['part_job_score']
    )

    # Normalize both to [1, 10]
    def rescale(x):
        return 1 + 9 * (x - np.min(x)) / (np.max(x) - np.min(x))

    df['stress_proxy'] = np.clip(rescale(Str_p) + np.random.normal(0, 0.2, len(df)), 1, 10)
    df['focus_proxy'] = np.clip(rescale(F_p) + np.random.normal(0, 0.2, len(df)), 1, 10)

    return df


def main():
    """Main execution function."""
    print("="*60)
    print("EXAM SCORE PREDICTION - ABLATION STUDY")
    print("="*60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Save processed data
    processed_data_path = Path('artifacts') / 'processed_data.csv'
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_data_path, index=False)
    print(f"\nProcessed data saved to: {processed_data_path}")
    
    # Initialize ablation study
    study = AblationStudy(
        data_path=str(processed_data_path),
        random_state=42
    )
    
    # Run all experiments
    print("\n" + "="*60)
    print("RUNNING ABLATION EXPERIMENTS")
    print("="*60)
    
    results = study.run_all_experiments(
        df=df,
        test_size=0.2,
        save_models=True,
        artifacts_dir='artifacts'
    )
    
    # Save results
    summary_df = study.save_results(output_dir='experiments')
    
    # Display top 10 results
    print("\n" + "="*60)
    print("TOP 10 EXPERIMENT RESULTS (by R²)")
    print("="*60)
    print(summary_df.head(10).to_string(index=False))
    
    # Save summary statistics
    stats_path = Path('experiments') / 'summary_statistics.txt'
    with open(stats_path, 'w') as f:
        f.write("ABLATION STUDY SUMMARY STATISTICS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total experiments: {len(results)}\n")
        f.write(f"Best R²: {summary_df['r2'].max():.4f}\n")
        f.write(f"Best MSE: {summary_df['mse'].min():.2f}\n")
        f.write(f"Best experiment: {summary_df.iloc[0]['experiment']}\n\n")
        f.write("Top 10 Results:\n")
        f.write(summary_df.head(10).to_string(index=False))
    
    print(f"\nSummary statistics saved to: {stats_path}")
    print("\n" + "="*60)
    print("ABLATION STUDY COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
