import streamlit as st
import pandas as pd
import numpy as np
import joblib


@st.cache_resource
def load_tools():
    return joblib.load('simple_fleet_model.pkl')

tools = load_tools()

st.set_page_config(page_title="Fleet Maintenance System", page_icon="⚙️", layout="centered")
st.title("⚙️ Enterprise Fleet Maintenance Predictor")


with st.form("prediction_form"):
    st.subheader("Vehicle Data")
    col1, col2 = st.columns(2)
    with col1:
        v_type = st.selectbox("Vehicle Type", ["Car", "Truck", "Bus"])
        usage = st.number_input("Usage Hours (Mileage)", min_value=0, value=5000)
        cost = st.number_input("Last Maintenance Cost ($)", min_value=0.0, value=150.0)
    with col2:
        brake = st.selectbox("Brake Condition", ["Good", "Fair", "Poor"])
        anomalies = st.selectbox("Anomalies Detected?", ["No", "Yes"])
        failure = st.selectbox("Past Failure History?", ["No", "Yes"])

    st.subheader("Sensor Readings")
    col3, col4 = st.columns(2)
    with col3:
        temp = st.number_input("Engine Temperature (°C)",min_value=85.0, value=90.0)
        tire = st.number_input("Tire Pressure (PSI)", value=35.0)
        vibration = st.slider("Vibration Level (mm/s)", 0.0, 10.0, 1.5)
    with col4:
        oil = st.slider("Oil Quality Score (0-100)", 0.0, 100.0, 85.0)
        battery = st.slider("Battery Voltage (V)", 10.0, 16.0, 13.5, step=0.1)

    submit = st.form_submit_button("Predict Maintenance Risk")


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
    if prob >= 0.5:
        st.error(f"🚨 **MAINTENANCE REQUIRED** (Risk Factor: {prob*100:.1f}%)")
    else:
        st.success(f"✅ **VEHICLE SAFE** (Risk Factor: {prob*100:.1f}%)")