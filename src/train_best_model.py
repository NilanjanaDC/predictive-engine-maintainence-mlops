"""
Train Best Model with Hyperparameter Tuning
Trains and tunes the best performing Random Forest model using GridSearchCV.
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    brier_score_loss,
    roc_curve
)


def load_data(data_dir='data'):
    """Load training and test data."""
    X_train = pd.read_csv(f'{data_dir}/X_train.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv').values.ravel()
    X_test = pd.read_csv(f'{data_dir}/X_test.csv')
    y_test = pd.read_csv(f'{data_dir}/y_test.csv').values.ravel()
    
    print(f"✓ Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model with comprehensive metrics."""
    preds = model.predict(X_test)
    
    # Handle Pipeline objects
    if hasattr(model, 'named_steps'):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = model.predict_proba(X_test)[:, 1]
    
    results = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "F2_Score": fbeta_score(y_test, preds, beta=2),
        "AUC": roc_auc_score(y_test, probs),
        "Brier_Score": brier_score_loss(y_test, probs)
    }
    
    return results, probs


def train_best_model(X_train, X_test, y_train, y_test):
    """Train and tune the best model (Random Forest with SMOTE)."""
    print("\n" + "="*60)
    print("TRAINING BEST MODEL (TUNED RANDOM FOREST WITH SMOTE)")
    print("="*60 + "\n")
    
    # Create pipeline with SMOTE + Random Forest
    pipeline = Pipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=5)),
        ("model", RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    
    # Hyperparameter grid for tuning
    param_grid = {
        "model__n_estimators": [300, 500],
        "model__max_depth": [6, 8, 12],
        "model__min_samples_leaf": [1, 3, 5]
    }
    
    print("Performing GridSearchCV for hyperparameter tuning...")
    print(f"Parameters to tune: {param_grid}\n")
    
    # Grid search with 5-fold cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="recall",
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print("\n" + "="*60)
    print(f"Best Parameters Found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best CV Score (Recall): {grid_search.best_score_:.4f}")
    print("="*60 + "\n")
    
    # Evaluate on test set
    metrics, probs = evaluate_model(best_model, X_test, y_test, "Best Model (Tuned RF + SMOTE)")
    
    print("\n" + "="*60)
    print("BEST MODEL PERFORMANCE ON TEST SET:")
    print("="*60)
    for metric, value in metrics.items():
        if metric != "Model":
            print(f"{metric:20s}: {value:.4f}")
    print("="*60 + "\n")
    
    return best_model, metrics, probs, best_params


def compute_roc_curve(model, X_test, y_test):
    """Compute ROC curve data."""
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    
    return fpr, tpr, thresholds


def save_model(model, model_path='models/best_model.joblib'):
    """Save the best model."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✓ Best model saved to {model_path}")


def save_metrics(metrics, metrics_path='reports/metrics.json'):
    """Save metrics to JSON."""
    import json
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    metrics_to_save = {k: float(v) for k, v in metrics.items() if k != "Model"}
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    print(f"✓ Metrics saved to {metrics_path}")


def main():
    """Run best model training pipeline."""
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Train and tune best model
    best_model, metrics, probs, best_params = train_best_model(X_train, X_test, y_train, y_test)
    
    # Compute ROC curve
    fpr, tpr, thresholds = compute_roc_curve(best_model, X_test, y_test)
    
    # Save model and metrics
    save_model(best_model)
    save_metrics(metrics)
    
    print("\n" + "="*60)
    print("Best model training completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
