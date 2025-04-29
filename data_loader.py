import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data(file_path):
    """
    Load the dataset from the given file path
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def explore_data(df):
    """
    Perform initial data exploration
    """
    print("\n=== Dataset Information ===")
    print(df.info())
    
    print("\n=== Basic Statistics ===")
    print(df.describe())
    
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    
    # Ensure img directory exists
    ensure_dir('img')
    
    # Plot correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join('img', 'correlation_matrix.png'))
    plt.close()
    
    # Plot target distribution
    if 'Outcome' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(x='Outcome', data=df)
        plt.title('Target Variable Distribution')
        plt.savefig(os.path.join('img', 'target_distribution.png'))
        plt.close()
        
        # Plot feature distributions by target
        features = df.columns.drop('Outcome')
        n_cols = 3
        n_rows = (len(features) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5*n_rows))
        for i, feature in enumerate(features, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.boxplot(x='Outcome', y=feature, data=df)
            plt.title(f'{feature} Distribution by Outcome')
        plt.tight_layout()
        plt.savefig(os.path.join('img', 'feature_distributions.png'))
        plt.close()

if __name__ == "__main__":
    # Test the functions
    df = load_data('diabetes.csv')
    if df is not None:
        explore_data(df) 