#!/bin/bash

# Deploy to Hugging Face Spaces using Git
# Usage: ./deploy_to_space.sh

SPACE_REPO="https://huggingface.co/spaces/nilanjanadevc/EnginePredictionML"
SPACE_DIR="./hf_space_temp"
HF_USERNAME="nilanjanadevc"

echo "🚀 Deploying to Hugging Face Spaces..."

# Clone the Space repo
if [ -d "$SPACE_DIR" ]; then
  rm -rf "$SPACE_DIR"
fi

git clone "$SPACE_REPO" "$SPACE_DIR"

# Copy deployment files to Space
cp deployment/app.py "$SPACE_DIR/app.py"
cp deployment/requirements.txt "$SPACE_DIR/requirements.txt"

# Create a README if it doesn't exist
if [ ! -f "$SPACE_DIR/README.md" ]; then
  cat > "$SPACE_DIR/README.md" << 'EOF'
---
title: Engine Predictive Maintenance
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.26.0
app_file: app.py
pinned: false
---

# Engine Predictive Maintenance

Predict engine failures before they happen using machine learning.

## Features
- Real-time sensor monitoring
- Batch prediction from CSV
- Risk assessment and alerts
- Model performance metrics

## How to use
1. Enter sensor readings manually or upload CSV
2. Get instant failure predictions
3. Export results for reporting

## Model
- Algorithm: Random Forest with SMOTE
- Accuracy: 92%+
- F2-Score: 0.657 (recall-optimized for failure detection)
EOF
fi

# Push to Space
cd "$SPACE_DIR"
git add .
git commit -m "Deploy Streamlit app and dependencies"
git push

echo "✅ Deployment complete!"
echo "🌐 Space: https://huggingface.co/spaces/nilanjanadevc/EnginePredictionML"

# Cleanup
cd ..
rm -rf "$SPACE_DIR"
