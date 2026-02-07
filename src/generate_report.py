"""
Generate Report Module
Creates a comprehensive markdown report of the model training and evaluation.
"""

import json
import os
import pandas as pd
from datetime import datetime


def load_results():
    """Load all evaluation results."""
    metrics = {}
    classification_report = {}
    confusion_matrix = {}
    threshold_analysis = {}
    
    reports_dir = 'reports'
    
    # Load metrics
    if os.path.exists(f'{reports_dir}/metrics.json'):
        with open(f'{reports_dir}/metrics.json', 'r') as f:
            metrics = json.load(f)
    
    # Load classification report
    if os.path.exists(f'{reports_dir}/classification_report.json'):
        with open(f'{reports_dir}/classification_report.json', 'r') as f:
            classification_report = json.load(f)
    
    # Load confusion matrix
    if os.path.exists(f'{reports_dir}/confusion_matrix.csv'):
        confusion_matrix = pd.read_csv(f'{reports_dir}/confusion_matrix.csv', index_col=0)
    
    # Load threshold analysis
    if os.path.exists(f'{reports_dir}/threshold_analysis.csv'):
        threshold_analysis = pd.read_csv(f'{reports_dir}/threshold_analysis.csv')
    
    return metrics, classification_report, confusion_matrix, threshold_analysis


def generate_report(metrics, classification_report, confusion_matrix, threshold_analysis):
    """Generate comprehensive markdown report."""
    
    report = f"""# Engine Predictive Maintenance Model Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

This report documents the performance of the predictive maintenance model trained to detect engine failures based on sensor readings. The model uses a Random Forest classifier with SMOTE-based imbalance handling to achieve optimal recall for failure detection.

### Key Metrics at a Glance
- **Accuracy**: {metrics.get('Accuracy', 'N/A'):.4f}
- **Precision**: {metrics.get('Precision', 'N/A'):.4f}
- **Recall**: {metrics.get('Recall', 'N/A'):.4f}
- **F1 Score**: {metrics.get('F1_Score', 'N/A'):.4f}
- **F2 Score**: {metrics.get('F2_Score', 'N/A'):.4f}
- **AUC-ROC**: {metrics.get('AUC', 'N/A'):.4f}
- **Brier Score**: {metrics.get('Brier_Score', 'N/A'):.4f}

---

## 1. Model Overview

### Architecture
- **Model Type**: Random Forest Classifier with SMOTE Oversampling
- **Training Approach**: GridSearchCV with 5-fold Cross-Validation
- **Optimization Metric**: Recall (Beta=2 for F2-Score)
- **Target Variable**: Engine Condition (0=Good, 1=Failure)

### Rationale
Random Forest was selected as the final model because it:
1. Provides stable predictions under sensor noise and extreme operating conditions
2. Supports explicit threshold tuning without retraining
3. Offers interpretable feature importance for maintenance teams
4. Balances recall and precision better than aggressive boosting approaches

---

## 2. Performance Metrics

### Overall Performance

| Metric | Value |
|--------|-------|
| Accuracy | {metrics.get('Accuracy', 'N/A'):.4f} |
| Precision | {metrics.get('Precision', 'N/A'):.4f} |
| Recall | {metrics.get('Recall', 'N/A'):.4f} |
| F1 Score | {metrics.get('F1_Score', 'N/A'):.4f} |
| F2 Score | {metrics.get('F2_Score', 'N/A'):.4f} |
| AUC-ROC | {metrics.get('AUC', 'N/A'):.4f} |
| PR-AUC | {metrics.get('PR_AUC', 'N/A'):.4f} |
| Brier Score | {metrics.get('Brier_Score', 'N/A'):.4f} |
| Specificity | {metrics.get('Specificity', 'N/A'):.4f} |

### Confusion Matrix

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | {metrics.get('True_Negatives', 0):.0f} | {metrics.get('False_Positives', 0):.0f} |
| **Actual Positive** | {metrics.get('False_Negatives', 0):.0f} | {metrics.get('True_Positives', 0):.0f} |

**Interpretation**:
- **True Negatives**: {metrics.get('True_Negatives', 0):.0f} (Correctly identified good engines)
- **False Positives**: {metrics.get('False_Positives', 0):.0f} (Good engines flagged as faulty - unnecessary maintenance)
- **False Negatives**: {metrics.get('False_Negatives', 0):.0f} (Failed engines missed by model - HIGH RISK)
- **True Positives**: {metrics.get('True_Positives', 0):.0f} (Correctly identified failing engines)

---

## 3. Threshold Analysis

The model's decision threshold can be tuned to optimize the recall-precision trade-off. Below is the performance at different thresholds:

"""
    
    if not threshold_analysis.empty:
        report += threshold_analysis.to_markdown(index=False)
    else:
        report += "Threshold analysis data not available."
    
    report += """

**Recommendation**: The default threshold of 0.5 balances recall and precision. However, for critical safety applications, a lower threshold (0.3-0.4) is recommended to maximize failure detection.

---

## 4. Model Interpretation

### Per-Class Performance

"""
    
    if classification_report:
        report += """
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
"""
        for class_label in ['0', '1']:
            if class_label in classification_report:
                cls_data = classification_report[class_label]
                report += f"| {class_label} | {cls_data.get('precision', 'N/A'):.4f} | {cls_data.get('recall', 'N/A'):.4f} | {cls_data.get('f1-score', 'N/A'):.4f} | {int(cls_data.get('support', 0))} |\n"
    
    report += """

### Safety Assessment

**Failure Detection Rate (Recall)**: 
- {:.4f} ({:.1f}% of actual failures detected)
- This means approximately 1 in every {:.0f} engine failures may go undetected.

**False Alarm Rate (1 - Precision)**:
- {:.4f} ({:.1f}% of flagged engines are false positives)
- This means approximately {:.1f}% of maintenance actions are unnecessary.

---

## 5. Business Impact

### Risk Analysis

1. **Missed Failures (False Negatives)**: {metrics.get('False_Negatives', 0):.0f}
   - Impact: Engine breakdowns, potential safety issues, high repair costs
   - Mitigation: Regular monitoring and maintenance intervals

2. **False Alarms (False Positives)**: {metrics.get('False_Positives', 0):.0f}
   - Impact: Unnecessary preventive maintenance, operational downtime
   - Mitigation: Use threshold tuning to reduce false positives

### ROI Metrics

- **Sensitivity (True Positive Rate)**: {metrics.get('Recall', 'N/A'):.4f}
- **Specificity (True Negative Rate)**: {metrics.get('Specificity', 'N/A'):.4f}
- **Positive Predictive Value**: {metrics.get('Precision', 'N/A'):.4f}

---

## 6. Recommendations

1. **Deploy with Caution**: The model shows competitive performance for predictive maintenance. Recommended for production with human-in-the-loop validation.

2. **Threshold Tuning**: Consider lowering the decision threshold to {:.2f} to increase recall if failure prevention is critical.

3. **Continuous Monitoring**: Track model performance in production and retrain quarterly with new failure data.

4. **Feature Importance Analysis**: Review which sensors contribute most to failure predictions and focus maintenance efforts there.

5. **Data Collection**: Continue collecting sensor data and actual failure incidents to improve model accuracy over time.

---

## 7. Model Limitations

1. **Imbalanced Data**: The dataset contains fewer failure cases, which may affect model generalization.
2. **Feature Interactions**: The model may miss complex interactions between sensors that indicate failure.
3. **Temporal Patterns**: Current features don't capture time-series degradation patterns.
4. **External Factors**: Environmental factors and usage patterns not captured in sensor data are ignored.

---

## Appendix: Technical Details

### Data Splits
- **Training Set**: 70% of data (with SMOTE oversampling)
- **Validation Set**: 10% of data
- **Test Set**: 20% of data

### Features Used
- Engine RPM
- Lube Oil Pressure & Temperature
- Coolant Pressure & Temperature
- Fuel Pressure
- Derived features (ratios, interactions, polynomial terms)

### Cross-Validation
- Method: 5-Fold Stratified Cross-Validation
- Optimization Metric: Recall

---

**Report Version**: 1.0  
**Model Version**: best_model.joblib  
**Status**: Production Ready (with monitoring)
""".format(
        metrics.get('Recall', 0),
        metrics.get('Recall', 0) * 100,
        1 / (1 - metrics.get('Recall', 0.5)) if metrics.get('Recall', 0) < 1 else float('inf'),
        1 - metrics.get('Precision', 0),
        (1 - metrics.get('Precision', 0)) * 100,
        (1 - metrics.get('Precision', 0)) * 100,
        0.5
    )
    
    return report


def save_report(report, output_path='reports/model_report.md'):
    """Save report to markdown file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Report saved to {output_path}")


def main():
    """Generate and save model report."""
    print("\n" + "="*70)
    print("GENERATING MODEL REPORT")
    print("="*70 + "\n")
    
    # Load results
    metrics, classification_report, confusion_matrix, threshold_analysis = load_results()
    
    if not metrics:
        print("⚠ No evaluation results found. Please run evaluate_model.py first.")
        return
    
    # Generate report
    report = generate_report(metrics, classification_report, confusion_matrix, threshold_analysis)
    
    # Save report
    save_report(report)
    
    print("\n" + "="*70)
    print("Report generation completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
