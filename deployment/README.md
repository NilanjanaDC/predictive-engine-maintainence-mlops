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

## 🎯 Features
- **Real-time Monitoring**: Enter sensor readings manually for instant predictions
- **Batch Processing**: Upload CSV files for bulk predictions
- **Risk Assessment**: Get risk level (Low/Medium/High) with confidence scores
- **Data Export**: Download predictions with timestamps for auditing

## 📊 Model Performance
- **Algorithm**: Random Forest with SMOTE (handles class imbalance)
- **Accuracy**: 92%+
- **F2-Score**: 0.657 (optimized for recall - prioritizes catching failures)
- **Training Data**: 1000+ engine sensor readings

## 🚀 Quick Start

### Manual Prediction
1. Enter 6 sensor readings (Lube Oil Pressure, Temperature, RPM, etc.)
2. Get instant failure probability and risk level
3. Review model confidence metrics

### Batch Prediction
1. Upload a CSV file with multiple engine readings
2. Get predictions for all rows
3. Download results with timestamps

## 📈 Sensor Inputs
- Lube Oil Pressure
- Lube Oil Temperature
- Coolant Temperature
- Engine RPM
- Fuel Pressure
- Coolant Pressure

## ⚙️ Technical Details
- **Model Format**: .joblib (scikit-learn native)
- **Data Processing**: Standardized features, engineered ratios
- **Inference**: <100ms per prediction
- **Deployment**: Hugging Face Spaces (auto-scaling)

## 📝 How It Works
1. Input sensors are standardized using training data statistics
2. Features are engineered (pressure ratios, temperature differences)
3. Random Forest classifier produces failure probability
4. F2-Score threshold (0.65) determines risk level classification

## 🔒 Privacy & Security
- All data processing happens in-browser
- No data is stored or logged
- Model weights are public and auditable
- Predictions are real-time only

## 🤝 Support
For issues or feedback, visit: https://github.com/your-org/predictive-engine-maintenance

---
*Built with scikit-learn, Streamlit, and Hugging Face Spaces*
