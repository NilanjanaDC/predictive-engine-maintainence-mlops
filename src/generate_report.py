"""
Generate Report Module
Creates a comprehensive markdown report of the model training and evaluation.
"""

import json
import os
import pandas as pd
from datetime import datetime


def safe_format(value, decimals=4):
    """Safely format a value as a float, returning 'N/A' if not numeric."""
    try:
        if value == 'N/A' or value is None:
            return 'N/A'
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return 'N/A'


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
    
    # Extract metrics safely
    accuracy = safe_format(metrics.get('Accuracy', 'N/A'))
    precision = safe_format(metrics.get('Precision', 'N/A'))
    recall = safe_format(metrics.get('Recall', 'N/A'))
    f1_score = safe_format(metrics.get('F1_Score', 'N/A'))
    f2_score = safe_format(metrics.get('F2_Score', 'N/A'))
    auc = safe_format(metrics.get('AUC', 'N/A'))
    brier = safe_format(metrics.get('Brier_Score', 'N/A'))
    
    report = f"""# Engine Predictive Maintenance Model Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

This report documents the performance of the predictive maintenance model trained to detect engine failures based on sensor readings. The model uses a Random Forest classifier with SMOTE-based imbalance handling to achieve optimal recall for failure detection.

### Key Metrics at a Glance
- **Accuracy**: {accuracy}
- **Precision**: {precision}
- **Recall**: {recall}
- **F1 Score**: {f1_score}
- **F2 Score**: {f2_score}
- **AUC-ROC**: {auc}
- **Brier Score**: {brier}

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
| Accuracy | {accuracy} |
| Precision | {precision} |
| Recall | {recall} |
| F1 Score | {f1_score} |
| F2 Score | {f2_score} |
| AUC-ROC | {auc} |
| PR-AUC | {safe_format(metrics.get('PR_AUC', 'N/A'))} |
| Brier Score | {brier} |
| Specificity | {safe_format(metrics.get('Specificity', 'N/A'))} |

### Confusion Matrix

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | {int(metrics.get('True_Negatives', 0))} | {int(metrics.get('False_Positives', 0))} |
| **Actual Positive** | {int(metrics.get('False_Negatives', 0))} | {int(metrics.get('True_Positives', 0))} |

**Interpretation**:
- **True Negatives**: {int(metrics.get('True_Negatives', 0))} (Correctly identified good engines)
- **False Positives**: {int(metrics.get('False_Positives', 0))} (Good engines flagged as faulty - unnecessary maintenance)
- **False Negatives**: {int(metrics.get('False_Negatives', 0))} (Failed engines missed by model - HIGH RISK)
- **True Positives**: {int(metrics.get('True_Positives', 0))} (Correctly identified failing engines)

---

## 3. Threshold Analysis

The model's decision threshold can be tuned to optimize the recall-precision trade-off. Below is the performance at different thresholds:

"""
    
    if isinstance(threshold_analysis, pd.DataFrame) and not threshold_analysis.empty:
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
                prec = safe_format(cls_data.get('precision', 'N/A'))
                rec = safe_format(cls_data.get('recall', 'N/A'))
                f1 = safe_format(cls_data.get('f1-score', 'N/A'))
                support = int(cls_data.get('support', 0))
                report += f"| {class_label} | {prec} | {rec} | {f1} | {support} |\n"
    
    report += """

### Safety Assessment

**Failure Detection Rate (Recall)**: 
- High recall is critical to avoid missed failures
- Current recall: """ + recall + """

**False Alarm Rate (1 - Precision)**:
- Lower is better for cost efficiency
- Current precision: """ + precision + """

---

## 5. Business Impact

### Risk Analysis

1. **Missed Failures (False Negatives)**: """ + str(int(metrics.get('False_Negatives', 0))) + """
   - Impact: Engine breakdowns, potential safety issues, high repair costs
   - Mitigation: Regular monitoring and maintenance intervals

2. **False Alarms (False Positives)**: """ + str(int(metrics.get('False_Positives', 0))) + """
   - Impact: Unnecessary preventive maintenance, operational downtime
   - Mitigation: Use threshold tuning to reduce false positives

### ROI Metrics

- **Sensitivity (True Positive Rate)**: """ + recall + """
- **Specificity (True Negative Rate)**: """ + safe_format(metrics.get('Specificity', 'N/A')) + """
- **Positive Predictive Value**: """ + precision + """

---

## 6. Recommendations

1. **Deploy with Caution**: The model shows competitive performance for predictive maintenance. Recommended for production with human-in-the-loop validation.

2. **Threshold Tuning**: Consider tuning the decision threshold based on your cost-benefit analysis of false positives vs false negatives.

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
"""
    
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
