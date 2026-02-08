"""
Feature Engineering Module
Handles data loading, preprocessing, and feature engineering for engine maintenance prediction.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(data_path='data/engine_data.csv'):
    """Load raw engine data."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"✓ Data loaded: {df.shape}")
    return df


def rename_columns(df):
    """Standardize column names."""
    df_renamed = df.copy()
    rename_mapping = {
        "Lub oil pressure": "Lube Oil Pressure",
        "lub oil temp": "Lube Oil Temperature",
        "Coolant temp": "Coolant Temperature",
        "Engine rpm": "Engine RPM",
        "Fuel pressure": "Fuel Pressure",
        "Coolant pressure": "Coolant Pressure"
    }
    df_renamed.rename(columns=rename_mapping, inplace=True)
    print(f"✓ Columns renamed")
    return df_renamed


def handle_missing_values(df):
    """Handle missing values in the dataset."""
    # Check and drop rows with missing values
    initial_rows = len(df)
    df_clean = df.dropna()
    removed_rows = initial_rows - len(df_clean)
    
    if removed_rows > 0:
        print(f"✓ Removed {removed_rows} rows with missing values")
    
    return df_clean


def remove_duplicates(df):
    """Remove duplicate rows."""
    initial_rows = len(df)
    df_unique = df.drop_duplicates()
    removed_rows = initial_rows - len(df_unique)
    
    if removed_rows > 0:
        print(f"✓ Removed {removed_rows} duplicate rows")
    
    return df_unique


def engineer_features(df):
    """Create additional features from existing sensor data."""
    df_features = df.copy()
    
    # Calculate sensor interactions and ratios
    sensor_columns = [col for col in df.columns if col != 'Engine Condition']
    
    # Add ratio features
    if 'Lube Oil Pressure' in df.columns and 'Coolant Pressure' in df.columns:
        df_features['Oil_Coolant_Pressure_Ratio'] = (
            df_features['Lube Oil Pressure'] / (df_features['Coolant Pressure'] + 1)
        )
    
    if 'Lube Oil Temperature' in df.columns and 'Coolant Temperature' in df.columns:
        df_features['Oil_Coolant_Temp_Diff'] = (
            df_features['Lube Oil Temperature'] - df_features['Coolant Temperature']
        )
    
    # Add rolling statistics for temporal patterns
    for col in sensor_columns:
        if col in df_features.columns:
            df_features[f'{col}_Squared'] = df_features[col] ** 2
    
    print(f"✓ Features engineered: {df_features.shape[1] - df.shape[1]} new features created")
    return df_features


def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    """Split data into train, validation, and test sets."""
    # Separate features and target
    X = df.drop('Engine Condition', axis=1)
    y = df['Engine Condition']
    
    # First split: 80% train+val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: Split temp into train and validation
    # val_size relative to temp (80%)
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=random_state, stratify=y_temp
    )
    
    print(f"✓ Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def standardize_features(X_train, X_val, X_test):
    """Standardize features using training data statistics."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print(f"✓ Features standardized")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def save_data(X_train, X_val, X_test, y_train, y_val, y_test, output_dir='data'):
    """Save processed data to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    X_val.to_csv(f'{output_dir}/X_val.csv', index=False)
    y_val.to_csv(f'{output_dir}/y_val.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)
    
    print(f"✓ Processed data saved to {output_dir}/")


def main():
    """Run feature engineering pipeline."""
    print("\n" + "="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60 + "\n")
    
    # Load data
    df = load_data()
    
    # Preprocessing
    df = rename_columns(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    
    # Feature engineering
    df = engineer_features(df)
    
    # Save engineered features
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/features_engineered.csv', index=False)
    print(f"✓ Engineered features saved: {df.shape}")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    # Standardize features
    X_train, X_val, X_test, scaler, = standardize_features(X_train, X_val, X_test)
    
    # Save processed data
    save_data(X_train, X_val, X_test, y_train, y_val, y_test)
    
    print("\n" + "="*60)
    print("Feature engineering completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
