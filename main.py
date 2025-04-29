from data_loader import load_data, explore_data
from data_cleaning import handle_missing_values, preprocess_data
from model_training import train_all_models
from model_evaluation import compare_models
import os
import joblib

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    # Create necessary directories
    ensure_dir('models')
    ensure_dir('img')
    ensure_dir('results')  # Add results directory for comparison file
    
    # Step 1: Load and explore data
    print("Step 1: Loading and exploring data...")
    df = load_data('diabetes.csv')
    explore_data(df)
    
    # Step 2: Clean and preprocess data
    print("\nStep 2: Cleaning and preprocessing data...")
    df_cleaned = handle_missing_values(df)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df_cleaned)
    
    # Step 3: Train models
    print("\nStep 3: Training models...")
    models, best_params = train_all_models(X_train, y_train)
    
    # Step 4: Evaluate models
    print("\nStep 4: Evaluating models...")
    results = compare_models(models, X_test, y_test)
    
    # Save model comparison results
    print("\nSaving model comparison results...")
    with open(os.path.join('results', 'model_comparison.txt'), 'w') as f:
        f.write("Model Comparison Results:\n\n")
        for model_name, metrics in results.items():
            f.write(f"=== {model_name} ===\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1 Score: {metrics['f1']:.4f}\n")
            f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n\n")
    
    # Find and save the best model based on accuracy
    best_model_name, best_model_metrics = max(results.items(), key=lambda x: x[1]['accuracy'])
    best_model = models[best_model_name]
    
    # Save the best model
    print(f"\nSaving best model: {best_model_name}")
    joblib.dump(best_model, os.path.join('models', 'best_model.pkl'))
    
    print("\nPipeline completed successfully!")
    print("\nBest performing model:")
    print(f"Model: {best_model_name}")
    print(f"Accuracy: {best_model_metrics['accuracy']:.4f}")
    print(f"F1 Score: {best_model_metrics['f1']:.4f}")
    print(f"ROC AUC: {best_model_metrics['roc_auc']:.4f}")

if __name__ == "__main__":
    main() 