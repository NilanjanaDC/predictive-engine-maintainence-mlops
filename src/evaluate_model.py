"""
Evaluate Model Module
Loads the trained model and evaluates it on test data, generating comprehensive metrics.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    brier_score_loss,
    roc_curve,
    confusion_matrix,
    classification_report
)


def load_data(data_dir='data'):
    """Load test data."""
    X_test = pd.read_csv(f'{data_dir}/X_test.csv')
    y_test = pd.read_csv(f'{data_dir}/y_test.csv').values.ravel()
    
    print(f"✓ Test data loaded: {X_test.shape}")
    return X_test, y_test


def load_model(model_path='models/best_model.joblib'):
    """Load trained model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = joblib.load(model_path)
    print(f"✓ Model loaded from {model_path}")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test data and compute metrics."""
    print("\n" + "="*70)
    print("EVALUATING MODEL ON TEST DATA")
    print("="*70 + "\n")
    
    # Get predictions and probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred)),
        "Recall": float(recall_score(y_test, y_pred)),
        "F1_Score": float(f1_score(y_test, y_pred)),
        "F2_Score": float(fbeta_score(y_test, y_pred, beta=2)),
        "AUC": float(roc_auc_score(y_test, y_proba)),
        "Brier_Score": float(brier_score_loss(y_test, y_proba))
    }
    
    # Display metrics
    print("Performance Metrics:")
    print("-" * 70)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:20s}: {metric_value:.4f}")
    print("-" * 70)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Compute classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
    return metrics, cm, class_report, (fpr, tpr, thresholds), y_pred, y_proba


def save_metrics(metrics, metrics_path='reports/metrics.json'):
    """Save metrics to JSON file."""
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Metrics saved to {metrics_path}")


def save_confusion_matrix(cm, cm_path='reports/confusion_matrix.csv'):
    """Save confusion matrix to CSV."""
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    
    cm_df = pd.DataFrame(cm, index=['Negative', 'Positive'], columns=['Predicted Negative', 'Predicted Positive'])
    cm_df.to_csv(cm_path)
    
    print(f"✓ Confusion matrix saved to {cm_path}")


def save_classification_report(class_report, report_path='reports/classification_report.json'):
    """Save classification report to JSON."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(class_report, f, indent=2)
    
    print(f"✓ Classification report saved to {report_path}")


def save_roc_curve_data(fpr, tpr, roc_path='reports/roc_curve.csv'):
    """Save ROC curve data to CSV."""
    os.makedirs(os.path.dirname(roc_path), exist_ok=True)
    
    roc_df = pd.DataFrame({
        'FPR': fpr,
        'TPR': tpr
    })
    roc_df.to_csv(roc_path, index=False)
    
    print(f"✓ ROC curve data saved to {roc_path}")


def main():
    """Main evaluation pipeline."""
    print("\n" + "="*70)
    print("MODEL EVALUATION PIPELINE")
    print("="*70 + "\n")
    
    try:
        # Load data and model
        X_test, y_test = load_data()
        model = load_model()
        
        # Evaluate model
        metrics, cm, class_report, (fpr, tpr, _), y_pred, y_proba = evaluate_model(model, X_test, y_test)
        
        # Save results
        save_metrics(metrics)
        save_confusion_matrix(cm)
        save_classification_report(class_report)
        save_roc_curve_data(fpr, tpr)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("="*70 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("Please ensure the model has been trained and saved.")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
