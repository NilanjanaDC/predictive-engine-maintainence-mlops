"""
Pipeline Verification Script
Validates that all pipeline components work correctly
"""

import os
import sys
import json

def check_file_exists(path, description):
    """Check if a file exists and report status."""
    if os.path.exists(path):
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description}: {path} NOT FOUND")
        return False

def check_directory_exists(path, description):
    """Check if a directory exists and report status."""
    if os.path.isdir(path):
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description}: {path} NOT FOUND")
        return False

def main():
    """Verify pipeline components."""
    print("\n" + "="*70)
    print("PIPELINE VERIFICATION")
    print("="*70 + "\n")
    
    all_good = True
    
    # Check source scripts
    print("Checking source scripts...")
    scripts = [
        ('src/feature_engineering.py', 'Feature Engineering Script'),
        ('src/train_models.py', 'Train Models Script'),
        ('src/train_best_model.py', 'Train Best Model Script'),
        ('src/evaluate_model.py', 'Evaluate Model Script'),
        ('src/generate_report.py', 'Generate Report Script'),
        ('push_to_huggingface.py', 'Push to HF Script'),
    ]
    
    for script_path, description in scripts:
        if not check_file_exists(script_path, description):
            all_good = False
    
    # Check data files
    print("\nChecking data files...")
    data_files = [
        ('data/engine_data.csv', 'Engine Data'),
        ('data/X_train.csv', 'Training Features'),
        ('data/y_train.csv', 'Training Labels'),
        ('data/X_test.csv', 'Test Features'),
        ('data/y_test.csv', 'Test Labels'),
    ]
    
    for data_path, description in data_files:
        if not check_file_exists(data_path, description):
            all_good = False
    
    # Check deployment files
    print("\nChecking deployment files...")
    deployment_files = [
        ('deployment/app.py', 'Streamlit App'),
        ('deployment/requirements.txt', 'Deployment Requirements'),
        ('deployment/README.md', 'Deployment README'),
    ]
    
    for deploy_path, description in deployment_files:
        if not check_file_exists(deploy_path, description):
            all_good = False
    
    # Check pipeline configuration
    print("\nChecking pipeline configuration...")
    if not check_file_exists('.github/workflows/pipeline.yml', 'GitHub Actions Pipeline'):
        all_good = False
    
    # Check directories
    print("\nChecking directories...")
    directories = [
        ('data', 'Data Directory'),
        ('src', 'Source Directory'),
        ('deployment', 'Deployment Directory'),
        ('.github/workflows', 'Workflows Directory'),
    ]
    
    for dir_path, description in directories:
        if not check_directory_exists(dir_path, description):
            all_good = False
    
    # Summary
    print("\n" + "="*70)
    if all_good:
        print("✓ All pipeline components are in place!")
        print("✓ Ready for CI/CD deployment")
    else:
        print("✗ Some components are missing. Please check above.")
        sys.exit(1)
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
