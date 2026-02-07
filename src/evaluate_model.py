"""
Model Evaluation Module
Evaluates the trained model and generates comprehensive metrics and visualizations.
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

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
    classification_report,
    precision_recall_curve,
    auc
)


def load_data(data_dir='data'):
    """Load test data."""
    X_test = pd.read_csv(f'{data_dir}/X_test.csv')
    y_test = pd.read_csv(f'{data_dir}/y_test.csv').values.ravel()
    
    print(f"✓ Test data loaded: {X_test.shape}")
    return X_test, y_test


def load_model(model_path='models/best_model.joblib'):
    """Load trained model."""
    model = joblib.load(model_path)
    print(f"✓ Model loaded from {model_path}")
    return model


def evaluate_model_comprehensive(model, X_test, y_test):
    """Comprehensive model evaluation with all metrics."""
    # Predictions
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1_Score": f1_score(y_test, preds),
        "F2_Score": fbeta_score(y_test, preds, beta=2),
        "AUC": roc_auc_score(y_test, probs),
        "Brier_Score": brier_score_loss(y_test, probs)
    }
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    
    # Derived metrics
    metrics["True_Negatives"] = int(tn)
    metrics["False_Positives"] = int(fp)
    metrics["False_Negatives"] = int(fn)
    metrics["True_Positives"] = int(tp)
    metrics["Specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    # Classification Report
    class_report = classification_report(y_test, preds, output_dict=True)
    
    # ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_test, probs)
    
    # Precision-Recall Curve
    precision, recall, pr_thresholds = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)
    metrics["PR_AUC"] = pr_auc
    
    return metrics, cm, class_report, fpr, tpr, roc_thresholds, precision, recall


def print_evaluation_report(metrics, cm, class_report):
    """Print comprehensive evaluation report."""
    print("\n" + "="*70)
    print("MODEL EVALUATION REPORT")
    print("="*70 + "\n")
    
    print("PERFORMANCE METRICS:")
    print("-" * 70)
    for metric, value in metrics.items():
        if metric not in ["True_Negatives", "False_Positives", "False_Negatives", "True_Positives"]:
            print(f"{metric:25s}: {value:.4f}")
    
    print("\n" + "-" * 70)
    print("CONFUSION MATRIX:")
    print("-" * 70)
    print(f"True Negatives:  {metrics['True_Negatives']:6d}")
    print(f"False Positives: {metrics['False_Positives']:6d}")
    print(f"False Negatives: {metrics['False_Negatives']:6d}")
    print(f"True Positives:  {metrics['True_Positives']:6d}")
    
    print("\n" + "-" * 70)
    print("CLASSIFICATION REPORT:")
    print("-" * 70)
    print(json.dumps(class_report, indent=2))
    
    print("\n" + "="*70 + "\n")


def generate_threshold_analysis(probs, y_test):
    """Analyze performance at different probability thresholds."""
    thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
    threshold_results = []
    
    for threshold in thresholds_to_test:
        preds = (probs >= threshold).astype(int)
        
        result = {
            "Threshold": threshold,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F2_Score": fbeta_score(y_test, preds, beta=2, zero_division=0)
        }
        threshold_results.append(result)
    
    threshold_df = pd.DataFrame(threshold_results)
    
    print("\nTHRESHOLD ANALYSIS:")
    print("-" * 70)
    print(threshold_df.to_string(index=False))
    print("-" * 70 + "\n")
    
    return threshold_df


def save_evaluation_results(metrics, cm, class_report, threshold_df, output_dir='reports'):
    """Save evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics_to_save = {k: float(v) if isinstance(v, (int, np.integer, float, np.floating)) else v 
                       for k, v in metrics.items()}
    
    with open(f'{output_dir}/metrics.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"✓ Metrics saved to {output_dir}/metrics.json")
    
    # Save confusion matrix
    cm_df = pd.DataFrame(cm, index=['Negative', 'Positive'], columns=['Predicted Negative', 'Predicted Positive'])
    cm_df.to_csv(f'{output_dir}/confusion_matrix.csv')
    print(f"✓ Confusion matrix saved to {output_dir}/confusion_matrix.csv")
    
    # Save classification report
    with open(f'{output_dir}/classification_report.json', 'w') as f:
        json.dump(class_report, f, indent=2)
    print(f"✓ Classification report saved to {output_dir}/classification_report.json")
    
    # Save threshold analysis
    threshold_df.to_csv(f'{output_dir}/threshold_analysis.csv', index=False)
    print(f"✓ Threshold analysis saved to {output_dir}/threshold_analysis.csv")


def main():
    """Run model evaluation pipeline."""
    print("\n" + "="*70)
    print("MODEL EVALUATION PIPELINE")
    print("="*70 + "\n")
    
    # Load data and model
    X_test, y_test = load_data()
    model = load_model()
    
    # Comprehensive evaluation
    metrics, cm, class_report, fpr, tpr, roc_thresholds, precision, recall = evaluate_model_comprehensive(
        model, X_test, y_test
    )
    
    # Print report
    probs = model.predict_proba(X_test)[:, 1]
    print_evaluation_report(metrics, cm, class_report)
    
    # Threshold analysis
    threshold_df = generate_threshold_analysis(probs, y_test)
    
    # Save results
    save_evaluation_results(metrics, cm, class_report, threshold_df)
    
    print("\n" + "="*70)
    print("Model evaluation completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
