import streamlit as st
import pandas as pd
import numpy as np
import joblib


@st.cache_resource
def load_tools():
    return joblib.load('simple_fleet_model.pkl')

tools = load_tools()

# Set up the page
st.set_page_config(page_title="Fleet Maintenance System", page_icon="⚙️", layout="wide")
st.title("⚙️ Fleet Maintenance Predictor")
st.markdown("#### Use this tool to predict the maintenance risk of your fleet vehicles with advanced analytics.")

st.divider()

# Tabs for better organization
tab1, tab2 = st.tabs(["🚗 Vehicle Information", "📊 Sensor Data"])

# Vehicle Information Tab
with tab1:
    st.subheader("Enter Vehicle Details")
    col1, col2 = st.columns(2)
    with col1:
        v_type = st.selectbox("Vehicle Type", ["Car", "Truck", "Bus"], help="Select the type of vehicle.")
        usage = st.number_input("Usage Hours (Mileage)", min_value=0, value=5000, help="Enter the total usage hours.")
        cost = st.number_input("Last Maintenance Cost ($)", min_value=0.0, value=150.0, help="Enter the cost of the last maintenance.")
    with col2:
        brake = st.selectbox("Brake Condition", ["Good", "Fair", "Poor"], help="Select the current brake condition.")
        anomalies = st.selectbox("Anomalies Detected?", ["No", "Yes"], help="Indicate if any anomalies were detected.")
        failure = st.selectbox("Past Failure History?", ["No", "Yes"], help="Indicate if there is a history of failures.")

# Sensor Data Tab
with tab2:
    st.subheader("Enter Sensor Readings")
    col3, col4 = st.columns(2)
    with col3:
        temp = st.number_input("Engine Temperature (°C)", min_value=85.0, value=90.0, help="Enter the engine temperature.")
        tire = st.number_input("Tire Pressure (PSI)", value=35.0, help="Enter the tire pressure.")
        vibration = st.slider("Vibration Level (mm/s)", 0.0, 10.0, 1.5, help="Set the vibration level.")
    with col4:
        oil = st.slider("Oil Quality Score (0-100)", 0.0, 100.0, 85.0, help="Set the oil quality score.")
        battery = st.slider("Battery Voltage (V)", 10.0, 16.0, 13.5, step=0.1, help="Set the battery voltage.")

st.divider()

# Submit Button
submit = st.button("🔍 Predict Maintenance Risk")

# Prediction logic
if submit:
    anom_val = 1 if anomalies == "Yes" else 0
    fail_val = 1 if failure == "Yes" else 0

    input_df = pd.DataFrame({
        'Usage_Hours': [usage], 'Engine_Temperature': [temp], 'Tire_Pressure': [tire],
        'Oil_Quality': [oil], 'Battery_Voltage': [battery], 'Vibration_Level': [vibration],
        'Maintenance_Cost': [cost], 'Anomalies_Detected': [anom_val], 'Failure_History': [fail_val],
        'Vehicle_Type': [v_type], 'Brake_Condition': [brake] 
    })

    input_num = input_df[tools['num_cols']]
    input_cat = input_df[tools['cat_cols']]

    num_filled = tools['num_imputer'].transform(input_num)
    cat_filled = tools['cat_imputer'].transform(input_cat)

    num_scaled = tools['scaler'].transform(num_filled)
    cat_encoded = tools['encoder'].transform(cat_filled)

    final_input = np.hstack((num_scaled, cat_encoded))

    prob = tools['model'].predict_proba(final_input)[0][1]

    st.divider()
    st.subheader("🔍 Prediction Result")
    if prob >= 0.5:
        st.error(f"🚨 **MAINTENANCE REQUIRED** (Risk Factor: {prob*100:.1f}%)")
    else:
        st.success(f"✅ **VEHICLE SAFE** (Risk Factor: {prob*100:.1f}%)")