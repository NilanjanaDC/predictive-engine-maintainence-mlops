import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Engine Predictive Maintenance System",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD MODEL FROM HUGGING FACE
# ============================================
@st.cache_resource
def load_model():
    """Load trained model from Hugging Face Hub"""
    try:
        model = hf_hub_download(
            repo_id="YOUR_HF_USERNAME/predictive-maintenance-model",
            filename="best_model.pkl"
        )
        return joblib.load(model)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ============================================
# FEATURE ENGINEERING FUNCTION
# ============================================
def engineer_features(df):
    """Apply physics-based feature engineering"""
    df_enhanced = df.copy()

    # Lubrication Stress Index
    df_enhanced['Lub_Stress_Index'] = df_enhanced['Lub oil pressure'] * df_enhanced['lub oil temp']

    # Thermal Efficiency
    df_enhanced['Thermal_Efficiency'] = df_enhanced['Coolant pressure'] / (df_enhanced['Coolant temp'] + 1e-5)

    # Power Load Index
    df_enhanced['Power_Load_Index'] = df_enhanced['Engine rpm'] * df_enhanced['Fuel pressure']

    return df_enhanced

# ============================================
# MAIN APP
# ============================================
st.title("🔧 Engine Predictive Maintenance System")
st.markdown("Real-time failure prediction using ML and physics-based features")

model = load_model()

if model is None:
    st.stop()

# ============================================
# SIDEBAR: INPUT METHOD
# ============================================
st.sidebar.header("⚙️ Input Configuration")
input_method = st.sidebar.radio(
    "Select input method:",
    ["📝 Manual Input", "📤 Upload CSV", "🔢 Batch Prediction"]
)

# ============================================
# MANUAL INPUT
# ============================================
if input_method == "📝 Manual Input":
    st.header("Manual Engine Sensor Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        engine_rpm = st.number_input("Engine RPM", min_value=0.0, max_value=3000.0, value=1000.0)
        lub_oil_pressure = st.number_input("Lub Oil Pressure (bar)", min_value=0.0, max_value=10.0, value=5.0)
        fuel_pressure = st.number_input("Fuel Pressure (bar)", min_value=0.0, max_value=10.0, value=3.5)

    with col2:
        coolant_pressure = st.number_input("Coolant Pressure (bar)", min_value=0.0, max_value=5.0, value=2.0)
        lub_oil_temp = st.number_input("Lub Oil Temp (°C)", min_value=0.0, max_value=150.0, value=80.0)
        coolant_temp = st.number_input("Coolant Temp (°C)", min_value=0.0, max_value=120.0, value=85.0)

    with col3:
        st.write("### Summary")
        st.info(f"✓ {6} sensor inputs ready")

    if st.button("🔍 Predict Engine Condition", key="predict_manual"):
        # Create dataframe
        input_data = pd.DataFrame({
            'Engine rpm': [engine_rpm],
            'Lub oil pressure': [lub_oil_pressure],
            'Fuel pressure': [fuel_pressure],
            'Coolant pressure': [coolant_pressure],
            'lub oil temp': [lub_oil_temp],
            'Coolant temp': [coolant_temp]
        })

        # Engineer features
        input_enhanced = engineer_features(input_data)

        # Make prediction
        prediction = model.predict(input_enhanced)[0]
        probability = model.predict_proba(input_enhanced)[0]

        # Display results
        st.success("✓ Prediction completed!")

        col_pred, col_prob = st.columns(2)

        with col_pred:
            if prediction == 0:
                st.metric("Status", "🟢 HEALTHY", delta="Normal Operation")
            else:
                st.metric("Status", "🔴 FAULTY", delta="Maintenance Required")

        with col_prob:
            st.metric("Confidence", f"{probability[prediction]*100:.2f}%")

        # Risk assessment
        st.subheader("📊 Risk Assessment")
        failure_risk = probability[1] * 100

        if failure_risk < 30:
            risk_level = "🟢 Low Risk"
        elif failure_risk < 70:
            risk_level = "🟡 Medium Risk"
        else:
            risk_level = "🔴 High Risk"

        st.write(f"Failure Risk: {risk_level} ({failure_risk:.2f}%)")

        # Feature importance for manual input
        st.subheader("🔍 Sensor Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Oil System:**")
            st.write(f"• Pressure: {lub_oil_pressure:.2f} bar")
            st.write(f"• Temp: {lub_oil_temp:.2f}°C")
        with col2:
            st.write("**Cooling System:**")
            st.write(f"• Pressure: {coolant_pressure:.2f} bar")
            st.write(f"• Temp: {coolant_temp:.2f}°C")
        with col3:
            st.write("**Engine Load:**")
            st.write(f"• RPM: {engine_rpm:.2f}")
            st.write(f"• Fuel: {fuel_pressure:.2f} bar")

# ============================================
# CSV UPLOAD
# ============================================
elif input_method == "📤 Upload CSV":
    st.header("Batch CSV Prediction")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview:")
        st.dataframe(df.head())

        if st.button("🔍 Predict All Rows"):
            df_enhanced = engineer_features(df)
            predictions = model.predict(df_enhanced)
            probabilities = model.predict_proba(df_enhanced)

            results_df = df.copy()
            results_df['Prediction'] = predictions
            results_df['Failure_Risk_%'] = probabilities[:, 1] * 100
            results_df['Status'] = results_df['Prediction'].apply(
                lambda x: "🟢 HEALTHY" if x == 0 else "🔴 FAULTY"
            )

            st.success("✓ Predictions completed!")
            st.dataframe(results_df)

            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# ============================================
# INFO SECTION
# ============================================
st.sidebar.markdown("---")
st.sidebar.header("ℹ️ About This Model")
st.sidebar.info("""
**Physics-Aware Predictive Maintenance System**

- **Training Data**: 19,535 engine observations
- **Features**: 9 (6 raw + 3 engineered)
- **Target**: Binary classification (Healthy/Faulty)
- **Primary Metric**: F2-Score (recall-focused)
- **Calibration**: Brier Score optimized

**Key Features:**
- Lubrication Stress Index
- Thermal Efficiency
- Power Load Index
""")

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Predictive Maintenance System v1.0")
