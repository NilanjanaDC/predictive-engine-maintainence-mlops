import os
import json
import joblib
from huggingface_hub import HfApi


def load_model_and_metadata(model_path='models/best_model.joblib', metrics_path='reports/metrics.json'):
    """Load model and metrics."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = joblib.load(model_path)
    print(f"✓ Model loaded from {model_path}")
    
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        print(f"✓ Metrics loaded from {metrics_path}")
    
    return model, metrics


def create_model_card(metrics):
    """Create a model card for the Hugging Face Hub."""
    
    # Extract metrics safely
    accuracy = metrics.get('Accuracy', 0)
    precision = metrics.get('Precision', 0)
    recall = metrics.get('Recall', 0)
    f1_score = metrics.get('F1_Score', 0)
    f2_score = metrics.get('F2_Score', 0)
    auc_score = metrics.get('AUC', 0)
    brier_score = metrics.get('Brier_Score', 0)
    
    model_card_content = f"""---
license: mit
language:
  - en
library_name: scikit-learn
tags:
  - predictive-maintenance
  - random-forest
  - binary-classification
  - engine-maintenance
datasets:
  - nasa-cmapss
metrics:
  - accuracy
  - f1
  - f2
  - roc-auc
---

# Engine Predictive Maintenance Model

## Model Overview
This is a **Tuned Random Forest Classifier** trained for predictive engine maintenance with SMOTE oversampling to handle class imbalance and achieve high recall for failure detection.

## Model Details
- **Model Type**: Random Forest Classifier with SMOTE Pipeline
- **Framework**: scikit-learn, imbalanced-learn
- **Task**: Binary Classification (Engine Condition: Good/Failing)
- **Input Features**: 14 engineered sensor features (RPM, pressure, temperature, etc.)
- **Output**: Probability of engine failure (0-1)

## Model Performance

### Test Set Metrics

| Metric | Score |
|--------|-------|
| Accuracy | {accuracy:.4f} |
| Precision | {precision:.4f} |
| Recall | {recall:.4f} |
| F1 Score | {f1_score:.4f} |
| F2 Score | {f2_score:.4f} |
| ROC-AUC | {auc_score:.4f} |
| Brier Score | {brier_score:.4f} |

## Key Insights
- **High Recall ({recall:.4f})**: Detects ~{recall*100:.0f}% of actual failures
- **Competitive Precision ({precision:.4f})**: ~{precision*100:.0f}% of predictions are correct
- **Strong AUC ({auc_score:.4f})**: Good discrimination between failure and non-failure cases

## Intended Use

This model is designed for:
- **Predictive Maintenance**: Identify engines at risk of failure before breakdown
- **Condition Monitoring**: Support data-driven maintenance decision-making
- **Fleet Management**: Optimize maintenance scheduling and resource allocation
- **Risk Assessment**: Provide failure probability scores for maintenance prioritization

## Limitations

- Trained on historical engine data with specific sensor configurations
- Performance may vary with new sensor types or operating conditions
- Model requires regular retraining with updated failure data
- Does not capture temporal degradation patterns (time-series)
- Assumes consistent sensor calibration and operating conditions

## Training Data

- **Dataset**: Engine Predictive Maintenance Dataset
- **Total Samples**: 19,581 engines
- **Training Samples**: 13,674 (70%)
- **Test Samples**: 3,907 (20%)
- **Features**: 14 engineered features (6 raw + 8 derived)
- **Class Distribution**: Imbalanced (Good: ~63%, Failure: ~37%)

## Training Procedure

1. Data preprocessing and feature engineering
2. Train-test split (70-20-10)
3. SMOTE oversampling on training data to handle class imbalance
4. Hyperparameter tuning via GridSearchCV with 5-fold cross-validation
5. Model evaluation on held-out test set

## Hyperparameters
- **n_estimators**: 400
- **max_depth**: 12
- **min_samples_leaf**: 4
- **SMOTE k_neighbors**: 5
- **Random state**: 42

## Recommendations

1. **Threshold Tuning**: Adjust decision threshold based on cost of false positives vs. false negatives
2. **Continuous Monitoring**: Track model performance in production and retrain quarterly with new data
3. **Feature Importance**: Use SHAP or feature importance analysis to identify critical sensors
4. **Ensemble Approaches**: Consider combining with other models (XGBoost, LightGBM) for robust predictions
5. **Domain Expertise**: Combine predictions with expert knowledge for final maintenance decisions

## Citation

If you use this model, please cite:

```
@misc{{predictive-maintenance-model-2026,
  title={{Engine Predictive Maintenance Model}},
  author={{GreatLearning Capstone Team}},
  year={{2026}},
  howpublished={{Hugging Face Hub}},
  url={{https://huggingface.co/models/nilanjanadevc/engine-predictive-maintenance-model}}
}}
```

## License

This model is released under the MIT License. See LICENSE file for details.

## Contact & Support

For questions or issues:
- GitHub: [Check repository](https://github.com/nilanjanadevc/predictive-engine-maintainence-mlops)
- Hugging Face: [@nilanjanadevc](https://huggingface.co/nilanjanadevc)
"""
    
    return model_card_content


def push_to_huggingface(model, model_card_content, repo_id, private=False):
    """Push model to Hugging Face Hub."""
    
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")
    
    print(f"\n{'='*70}")
    print("PUSHING MODEL TO HUGGING FACE HUB")
    print(f"{'='*70}\n")
    
    print(f"Repository ID: {repo_id}")
    print(f"Private: {private}")
    
    try:
        # Initialize API
        api = HfApi(token=hf_token)
        
        # Create repository if it doesn't exist
        print("\nCreating/accessing repository...")
        repo_url = api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True
        )
        print(f"✓ Repository ready: {repo_url}")
        
        # Save model locally
        local_model_path = "temp_model.joblib"
        joblib.dump(model, local_model_path)
        
        # Upload model file
        print("\nUploading model file...")
        api.upload_file(
            path_or_fileobj=local_model_path,
            path_in_repo="model.joblib",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload trained Random Forest model"
        )
        print(f"✓ Model file uploaded")
        
        # Upload model card
        print("Uploading model card...")
        api.upload_file(
            path_or_fileobj=model_card_content.encode('utf-8'),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model documentation"
        )
        print(f"✓ Model card uploaded")
        
        # Clean up temporary file
        if os.path.exists(local_model_path):
            os.remove(local_model_path)
        
        print(f"\n✓ Successfully pushed model to {repo_url}")
        
        return repo_url
        
    except Exception as e:
        print(f"\n✗ Error pushing to Hugging Face: {str(e)}")
        raise


def main():
    """Main push to Hugging Face pipeline."""
    print("\n" + "="*70)
    print("PUSHING MODEL TO HUGGING FACE HUB")
    print("="*70 + "\n")
    
    try:
        # Load model and metrics
        model, metrics = load_model_and_metadata()
        
        # Create model card
        model_card_content = create_model_card(metrics)
        
        # Repository details
        repo_id = os.environ.get('HF_REPO_ID', 'nilanjanadevc/engine-predictive-maintenance-model')
        
        # Push to Hugging Face
        repo_url = push_to_huggingface(
            model=model,
            model_card_content=model_card_content,
            repo_id=repo_id,
            private=False
        )
        
        print("\n" + "="*70)
        print("MODEL SUCCESSFULLY DEPLOYED TO HUGGING FACE HUB")
        print(f"Access your model at: {repo_url}")
        print("="*70 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("Please ensure the model has been trained and saved.")
    except ValueError as e:
        print(f"\n✗ Configuration Error: {e}")
        print("Please set the HF_TOKEN environment variable before running this script.")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
