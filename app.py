import streamlit as st
import pandas as pd
import joblib

# Load the Decision Tree Pipeline
@st.cache_resource
def load_model():
    return joblib.load('decision_tree_pipeline.pkl')

model = load_model()

st.set_page_config(page_title="Maintenance System", page_icon="⚙️", layout="centered")
st.title("⚙️ Vehicle Maintenance Prediction System")

# UI Form
with st.form("prediction_form"):
    st.subheader("Vehicle Data")
    col1, col2 = st.columns(2)
    with col1:
        usage = st.number_input("Usage Hours (Mileage)", min_value=0, value=5000)
        cost = st.number_input("Last Maintenance Cost ($)", min_value=0.0, value=150.0)
        v_type = st.selectbox("Vehicle Type", ["Truck", "Van"])
    with col2:
        temp = st.number_input("Engine Temperature (°C)", value=90.0)
        tire = st.number_input("Tire Pressure (PSI)", value=35.0)
        oil = st.slider("Oil Quality Score (0-100)", 0.0, 100.0, 85.0)
        
    brake = st.selectbox("Brake Condition", ["Good", "Fair", "Poor"])
    anomalies = st.selectbox("Anomalies Detected?", ["No", "Yes"])
    failure = st.selectbox("Past Failure History?", ["No", "Yes"])

    submit = st.form_submit_button("Predict Maintenance Risk")

# Prediction Logic
if submit:
    # Map Yes/No to 1/0
    anom_val = 1 if anomalies == "Yes" else 0
    fail_val = 1 if failure == "Yes" else 0

    # Create a DataFrame matching the training features exactly
    input_df = pd.DataFrame({
        'Usage_Hours': [usage], 'Engine_Temperature': [temp], 'Tire_Pressure': [tire],
        'Oil_Quality': [oil], 'Maintenance_Cost': [cost], 'Vehicle_Type': [v_type],
        'Brake_Condition': [brake], 'Anomalies_Detected': [anom_val], 'Failure_History': [fail_val]
    })

    # Predict Risk
    prob = model.predict_proba(input_df)[0][1]
    
    st.divider()
    if prob >= 0.5:
        st.error(f"🚨 **MAINTENANCE REQUIRED** (Risk Factor: {prob*100:.1f}%)")
    else:
        st.success(f"✅ **VEHICLE SAFE** (Risk Factor: {prob*100:.1f}%)")
        