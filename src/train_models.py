"""
Train Baseline Models
Trains multiple baseline classification models for engine maintenance prediction.
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    brier_score_loss
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
    
    return results


def train_baseline_models(X_train, X_test, y_train, y_test):
    """Train and evaluate baseline models."""
    print("\n" + "="*60)
    print("TRAINING BASELINE MODELS")
    print("="*60 + "\n")
    
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Bagging": BaggingClassifier(random_state=42, n_jobs=-1),
        "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...", end=" ")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        trained_models[name] = model
        print(f"✓ F2-Score: {metrics['F2_Score']:.4f}")
    
    results_df = pd.DataFrame(results)
    
    # Save results
    os.makedirs('reports', exist_ok=True)
    results_df.to_csv('reports/baseline_models_results.csv', index=False)
    
    print("\n" + "="*60)
    print("Baseline Models Comparison:")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60 + "\n")
    
    return trained_models, results_df


def save_models(trained_models, model_dir='models'):
    """Save trained models to disk."""
    os.makedirs(model_dir, exist_ok=True)
    
    for name, model in trained_models.items():
        model_path = f'{model_dir}/{name.lower().replace(" ", "_")}.pkl'
        joblib.dump(model, model_path)
        print(f"✓ Saved {name} model")
    
    print(f"✓ All baseline models saved to {model_dir}/")


def main():
    """Run baseline model training pipeline."""
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Train baseline models
    trained_models, results_df = train_baseline_models(X_train, X_test, y_train, y_test)
    
    # Save models
    save_models(trained_models)
    
    print("\n" + "="*60)
    print("Baseline model training completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
