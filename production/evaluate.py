"""
Evaluation and Analysis Script

Load experiment results and generate comprehensive analysis.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


def load_results(experiments_dir: str = 'experiments'):
    """Load experiment results."""
    exp_path = Path(experiments_dir)
    
    # Load JSON results
    json_path = exp_path / 'ablation_results.json'
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    # Load summary CSV
    csv_path = exp_path / 'ablation_summary.csv'
    summary_df = pd.read_csv(csv_path)
    
    return results, summary_df


def analyze_scaler_impact(summary_df):
    """Analyze impact of different scalers."""
    print("\n" + "="*60)
    print("SCALER IMPACT ANALYSIS")
    print("="*60)
    
    scaler_results = summary_df.groupby('scaler').agg({
        'r2': ['mean', 'std', 'max'],
        'mse': ['mean', 'std', 'min'],
        'training_time': 'mean'
    }).round(4)
    
    print(scaler_results)
    
    return scaler_results


def analyze_encoder_impact(summary_df):
    """Analyze impact of different encoders."""
    print("\n" + "="*60)
    print("ENCODER IMPACT ANALYSIS")
    print("="*60)
    
    encoder_results = summary_df.groupby('encoder').agg({
        'r2': ['mean', 'std', 'max'],
        'mse': ['mean', 'std', 'min'],
        'training_time': 'mean'
    }).round(4)
    
    print(encoder_results)
    
    return encoder_results


def analyze_model_impact(summary_df):
    """Analyze impact of different models."""
    print("\n" + "="*60)
    print("MODEL IMPACT ANALYSIS")
    print("="*60)
    
    model_results = summary_df.groupby('model').agg({
        'r2': ['mean', 'std', 'max'],
        'mse': ['mean', 'std', 'min'],
        'mae': ['mean', 'std', 'min'],
        'training_time': 'mean'
    }).round(4)
    
    print(model_results)
    
    return model_results


def create_visualizations(summary_df, output_dir='experiments/plots'):
    """Create visualization plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    sns.set_style('whitegrid')
    
    # 1. R² by Model Type
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=summary_df, x='model', y='r2')
    plt.title('R² Score by Model Type', fontsize=14, weight='bold')
    plt.xlabel('Model Type')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path / 'r2_by_model.png', dpi=300)
    print(f"Saved: {output_path / 'r2_by_model.png'}")
    plt.close()
    
    # 2. MSE by Scaler
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=summary_df, x='scaler', y='mse')
    plt.title('MSE by Scaler Type', fontsize=14, weight='bold')
    plt.xlabel('Scaler Type')
    plt.ylabel('Mean Squared Error')
    plt.tight_layout()
    plt.savefig(output_path / 'mse_by_scaler.png', dpi=300)
    print(f"Saved: {output_path / 'mse_by_scaler.png'}")
    plt.close()
    
    # 3. Training Time vs R²
    plt.figure(figsize=(10, 6))
    for model in summary_df['model'].unique():
        model_data = summary_df[summary_df['model'] == model]
        plt.scatter(model_data['training_time'], model_data['r2'], 
                   label=model, alpha=0.6, s=100)
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('R² Score')
    plt.title('Training Time vs Performance', fontsize=14, weight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'time_vs_performance.png', dpi=300)
    print(f"Saved: {output_path / 'time_vs_performance.png'}")
    plt.close()
    
    # 4. Top 10 Experiments
    top10 = summary_df.nlargest(10, 'r2')
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top10, x='r2', y='experiment')
    plt.title('Top 10 Experiments by R² Score', fontsize=14, weight='bold')
    plt.xlabel('R² Score')
    plt.ylabel('Experiment Name')
    plt.tight_layout()
    plt.savefig(output_path / 'top10_experiments.png', dpi=300)
    print(f"Saved: {output_path / 'top10_experiments.png'}")
    plt.close()
    
    # 5. Encoder comparison
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=summary_df, x='encoder', y='r2')
    plt.title('R² Distribution by Encoder Type', fontsize=14, weight='bold')
    plt.xlabel('Encoder Type')
    plt.ylabel('R² Score')
    plt.tight_layout()
    plt.savefig(output_path / 'r2_by_encoder.png', dpi=300)
    print(f"Saved: {output_path / 'r2_by_encoder.png'}")
    plt.close()
    
    print(f"\n✅ All plots saved to {output_path}/")


def generate_report(summary_df, output_path='experiments/analysis_report.txt'):
    """Generate comprehensive text report."""
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("ABLATION STUDY ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*60 + "\n")
        f.write(f"Total Experiments: {len(summary_df)}\n")
        f.write(f"Best R²: {summary_df['r2'].max():.4f}\n")
        f.write(f"Worst R²: {summary_df['r2'].min():.4f}\n")
        f.write(f"Mean R²: {summary_df['r2'].mean():.4f} ± {summary_df['r2'].std():.4f}\n")
        f.write(f"Best MSE: {summary_df['mse'].min():.2f}\n")
        f.write(f"Worst MSE: {summary_df['mse'].max():.2f}\n\n")
        
        # Best configuration
        best_exp = summary_df.iloc[0]
        f.write("BEST CONFIGURATION\n")
        f.write("-"*60 + "\n")
        f.write(f"Experiment: {best_exp['experiment']}\n")
        f.write(f"Model: {best_exp['model']}\n")
        f.write(f"Scaler: {best_exp['scaler']}\n")
        f.write(f"Encoder: {best_exp['encoder']}\n")
        f.write(f"R²: {best_exp['r2']:.4f}\n")
        f.write(f"MSE: {best_exp['mse']:.2f}\n")
        f.write(f"MAE: {best_exp['mae']:.2f}\n")
        f.write(f"RMSE: {best_exp['rmse']:.2f}\n")
        f.write(f"Training Time: {best_exp['training_time']:.2f}s\n\n")
        
        # Top 10
        f.write("TOP 10 EXPERIMENTS\n")
        f.write("-"*60 + "\n")
        top10 = summary_df.head(10)[['experiment', 'model', 'scaler', 'encoder', 'r2', 'mse']]
        f.write(top10.to_string(index=False))
        f.write("\n\n")
        
        # Model comparison
        f.write("MODEL COMPARISON\n")
        f.write("-"*60 + "\n")
        model_comp = summary_df.groupby('model')['r2'].agg(['mean', 'std', 'max']).round(4)
        f.write(model_comp.to_string())
        f.write("\n\n")
        
        # Scaler comparison
        f.write("SCALER COMPARISON\n")
        f.write("-"*60 + "\n")
        scaler_comp = summary_df.groupby('scaler')['r2'].agg(['mean', 'std', 'max']).round(4)
        f.write(scaler_comp.to_string())
        f.write("\n\n")
        
        # Key insights
        f.write("KEY INSIGHTS\n")
        f.write("-"*60 + "\n")
        
        best_model = summary_df.groupby('model')['r2'].mean().idxmax()
        best_scaler = summary_df.groupby('scaler')['r2'].mean().idxmax()
        best_encoder = summary_df.groupby('encoder')['r2'].mean().idxmax()
        
        f.write(f"1. Best Model (avg): {best_model}\n")
        f.write(f"2. Best Scaler (avg): {best_scaler}\n")
        f.write(f"3. Best Encoder (avg): {best_encoder}\n")
        
        # Performance tier
        excellent = len(summary_df[summary_df['r2'] >= 0.95])
        good = len(summary_df[(summary_df['r2'] >= 0.90) & (summary_df['r2'] < 0.95)])
        moderate = len(summary_df[(summary_df['r2'] >= 0.80) & (summary_df['r2'] < 0.90)])
        poor = len(summary_df[summary_df['r2'] < 0.80])
        
        f.write(f"\n4. Performance Distribution:\n")
        f.write(f"   - Excellent (R² ≥ 0.95): {excellent} experiments\n")
        f.write(f"   - Good (0.90 ≤ R² < 0.95): {good} experiments\n")
        f.write(f"   - Moderate (0.80 ≤ R² < 0.90): {moderate} experiments\n")
        f.write(f"   - Poor (R² < 0.80): {poor} experiments\n")
    
    print(f"\n✅ Report saved to {output_path}")


def main():
    """Main evaluation function."""
    print("="*60)
    print("EXPERIMENT EVALUATION AND ANALYSIS")
    print("="*60)
    
    # Load results
    results, summary_df = load_results()
    
    print(f"\nLoaded {len(summary_df)} experiment results")
    
    # Run analyses
    analyze_scaler_impact(summary_df)
    analyze_encoder_impact(summary_df)
    analyze_model_impact(summary_df)
    
    # Create visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    create_visualizations(summary_df)
    
    # Generate report
    print("\n" + "="*60)
    print("GENERATING REPORT")
    print("="*60)
    generate_report(summary_df)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
