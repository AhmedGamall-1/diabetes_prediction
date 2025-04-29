from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a single model and return metrics
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    # Ensure img directory exists
    ensure_dir('img')
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join('img', f'confusion_matrix_{model_name}.png'))
    plt.close()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {metrics["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.savefig(os.path.join('img', f'roc_curve_{model_name}.png'))
    plt.close()
    
    # Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.savefig(os.path.join('img', f'pr_curve_{model_name}.png'))
    plt.close()
    
    # Print classification report
    print(f"\n=== Classification Report for {model_name} ===")
    print(classification_report(y_test, y_pred))
    
    return metrics

def compare_models(models, X_test, y_test):
    """
    Compare multiple models and return their metrics
    """
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        results[name] = evaluate_model(model, X_test, y_test, name)
    
    # Plot comparison of metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    x = np.arange(len(metrics))
    width = 0.15  # Adjusted for more models
    
    plt.figure(figsize=(15, 8))
    for i, (name, result) in enumerate(results.items()):
        values = [result[metric] for metric in metrics]
        plt.bar(x + i*width, values, width, label=name)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.xticks(x + width*len(models)/2, metrics)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join('img', 'model_comparison.png'))
    plt.close()
    
    return results

def save_results(results):
    """
    Save evaluation results to a file
    """
    ensure_dir('results')
    with open(os.path.join('results', 'model_evaluation_results.txt'), 'w') as f:
        for model_name, metrics in results.items():
            f.write(f"\n=== {model_name} ===\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n") 