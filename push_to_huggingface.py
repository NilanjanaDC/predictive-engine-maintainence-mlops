"""
Push Model to Hugging Face Hub
Uploads the trained model to Hugging Face Model Hub for deployment.
"""

import os
import json
import joblib
from huggingface_hub import HfApi, ModelCard
from pathlib import Path


def load_model_and_metadata(model_path='models/best_model.pkl', metrics_path='reports/metrics.json'):
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
    
    model_card_content = f"""---
license: mit
datasets:
- engine-predictive-maintenance-processed
metrics:
- accuracy
- precision
- recall
- f1
- roc_auc
task_ids:
- tabular-classification
---

# Engine Predictive Maintenance Model

## Model Description

This is a Random Forest classifier trained to predict engine failures based on sensor readings. The model uses SMOTE oversampling to handle class imbalance and achieves high recall for failure detection, which is critical in a maintenance context.

### Model Details

- **Model Type**: Random Forest with SMOTE Pipeline
- **Framework**: scikit-learn, imbalanced-learn
- **Task**: Binary Classification (Engine Condition: Good/Failing)
- **Input Features**: Engine sensors (RPM, pressure, temperature, etc.)
- **Output**: Probability of engine failure

## Model Performance

### Test Set Metrics

| Metric | Value |
|--------|-------|
| Accuracy | {metrics.get('Accuracy', 'N/A'):.4f} |
| Precision | {metrics.get('Precision', 'N/A'):.4f} |
| Recall | {metrics.get('Recall', 'N/A'):.4f} |
| F1 Score | {metrics.get('F1_Score', 'N/A'):.4f} |
| F2 Score | {metrics.get('F2_Score', 'N/A'):.4f} |
| AUC-ROC | {metrics.get('AUC', 'N/A'):.4f} |
| Brier Score | {metrics.get('Brier_Score', 'N/A'):.4f} |

## Intended Use

This model is designed for:
- **Predictive Maintenance**: Identify engines at risk of failure before breakdown
- **Condition Monitoring**: Support data-driven maintenance decision-making
- **Fleet Management**: Optimize maintenance scheduling and resource allocation

## Limitations

- Trained on historical engine data with specific sensor configurations
- Performance may vary with new sensor types or operating conditions
- Model requires regular retraining with updated failure data
- Does not capture temporal degradation patterns (time-series)

## Training Data

- **Dataset**: Engine Predictive Maintenance Dataset
- **Total Samples**: ~19,000 engines
- **Training Samples**: ~13,300 (70%)
- **Test Samples**: ~3,800 (20%)
- **Features**: 8 continuous sensor variables + derived features
- **Class Distribution**: Imbalanced (Good: ~63%, Failure: ~37%)

## Training Procedure

1. Data preprocessing and feature engineering
2. Train-test split (70-20-10)
3. SMOTE oversampling on training data
4. Hyperparameter tuning via GridSearchCV
5. Evaluation on held-out test set

## Evaluation Results

The model achieves:
- **High Recall ({:.4f})**: Detects ~{:.0f}% of actual failures
- **Competitive Precision ({:.4f})**: ~{:.0f}% of predictions are correct
- **Strong AUC ({:.4f})**: Good discrimination between classes

## Recommendations

1. **Threshold Tuning**: Adjust decision threshold based on maintenance cost vs. failure cost trade-off
2. **Continuous Monitoring**: Track model performance in production and retrain quarterly
3. **Feature Importance**: Use model to identify critical sensors for maintenance teams
4. **Ensemble Approaches**: Consider combining with other models for robust predictions

## Citation

If you use this model, please cite:

```
@model{{engine_maintenance_rf_2024,
  title={{Engine Predictive Maintenance Model}},
  author={{GreatLearning MLOps Capstone}},
  year={{2024}},
  note={{Random Forest with SMOTE for failure prediction}}
}}
```

## License

This model is released under the MIT License. See LICENSE file for details.

## Acknowledgments

Trained as part of the GreatLearning MLOps Capstone project on predictive engine maintenance.
""".format(
        metrics.get('Recall', 0),
        metrics.get('Recall', 0) * 100,
        metrics.get('Precision', 0),
        metrics.get('Precision', 0) * 100,
        metrics.get('AUC', 0)
    )
    
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
        local_model_path = "temp_model.pkl"
        joblib.dump(model, local_model_path)
        
        # Upload model file
        print("\nUploading model file...")
        api.upload_file(
            path_or_fileobj=local_model_path,
            path_in_repo="model.pkl",
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
        # This should be customized with actual HF username
        repo_id = os.environ.get('HF_REPO_ID', 'engine-maintenance-predictor')
        
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
