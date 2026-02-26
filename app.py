import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page Config
st.set_page_config(
    page_title="Fleet Maintenance Pro",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    :root {
        --primary-color: #0066cc;
        --danger-color: #ff4444;
        --success-color: #00cc44;
        --warning-color: #ffaa00;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0066cc 0%, #004499 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0, 102, 204, 0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5em;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        margin: 10px 0 0 0;
        font-size: 1.1em;
        opacity: 0.95;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #0066cc;
        margin: 10px 0;
    }
    
    .section-header {
        font-size: 1.3em;
        font-weight: 700;
        color: #0066cc;
        margin: 20px 0 15px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #0066cc;
    }
    
    .input-section {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        margin-bottom: 15px;
    }
    
    .result-safe {
        background: linear-gradient(135deg, #00cc44 0%, #009933 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 204, 68, 0.3);
    }
    
    .result-maintenance {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255, 68, 68, 0.3);
    }
    
    .result-warning {
        background: linear-gradient(135deg, #ffaa00 0%, #ff8800 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255, 170, 0, 0.3);
    }
    
    .result-title {
        font-size: 2em;
        font-weight: 700;
        margin-bottom: 10px;
    }
    
    .result-subtitle {
        font-size: 1.3em;
        opacity: 0.95;
    }
    
    .gauge-container {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
    
    .info-box {
        background: #f0f4ff;
        border-left: 4px solid #0066cc;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    .tab-header {
        background: linear-gradient(90deg, #0066cc 0%, #004499 100%);
        color: white;
        padding: 15px;
        border-radius: 8px 8px 0 0;
        font-weight: 700;
    }
    
    .divider {
        margin: 25px 0;
        border: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #0066cc, transparent);
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_tools():
    return joblib.load('simple_fleet_model.pkl')

tools = load_tools()

# Header Section
st.markdown("""
<div class="main-header">
    <h1>🚗 Fleet Maintenance Pro</h1>
    <p>AI-Powered Predictive Maintenance System</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("### 📋 Navigation")
    tab_selection = st.radio(
        "Select Mode:",
        ["🔍 Predict", "ℹ️ Information", "⚡ Quick Tips"],
        label_visibility="collapsed"
    )

if tab_selection == "🔍 Predict":
    # Main Prediction Interface
    st.markdown('<div class="section-header">🚙 Vehicle Diagnosis</div>', unsafe_allow_html=True)
    
    with st.form("prediction_form", clear_on_submit=False):
        # Vehicle Information Section
        st.markdown('<div class="tab-header">📊 Vehicle Information</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            v_type = st.selectbox(
                "Vehicle Type",
                ["Car", "Truck", "Bus"],
                help="Select the type of vehicle"
            )
        with col2:
            brake = st.selectbox(
                "Brake Condition",
                ["Good", "Fair", "Poor"],
                help="Current brake system status"
            )
        with col3:
            failure = st.selectbox(
                "Past Failure History?",
                ["No", "Yes"],
                help="Has this vehicle had maintenance issues before?"
            )
        
        st.divider()
        
        # Usage & Cost Section
        st.markdown('<div class="tab-header">⛽ Usage & Maintenance</div>', unsafe_allow_html=True)
        
        col4, col5, col6 = st.columns(3)
        with col4:
            usage = st.number_input(
                "Usage Hours (Mileage)",
                min_value=0,
                value=5000,
                step=100,
                help="Total hours/miles the vehicle has been in use"
            )
        with col5:
            cost = st.number_input(
                "Last Maintenance Cost ($)",
                min_value=0.0,
                value=150.0,
                step=10.0,
                help="Cost of the last maintenance service"
            )
        with col6:
            anomalies = st.selectbox(
                "Anomalies Detected?",
                ["No", "Yes"],
                help="Any unusual sensor readings detected?"
            )
        
        st.divider()
        
        # Engine Sensor Readings
        st.markdown('<div class="tab-header">🌡️ Engine Sensors</div>', unsafe_allow_html=True)
        
        col7, col8 = st.columns(2)
        with col7:
            temp = st.number_input(
                "Engine Temperature (°C)",
                min_value=85.0,
                value=90.0,
                step=1.0,
                help="Current engine operating temperature"
            )
        with col8:
            tire = st.number_input(
                "Tire Pressure (PSI)",
                min_value=20.0,
                value=35.0,
                step=0.5,
                help="Average tire pressure across all wheels"
            )
        
        st.divider()
        
        # Performance Metrics
        st.markdown('<div class="tab-header">📈 Performance Metrics</div>', unsafe_allow_html=True)
        
        col9, col10 = st.columns(2)
        with col9:
            vibration = st.slider(
                "Vibration Level (mm/s)",
                0.0, 10.0, 1.5,
                step=0.1,
                help="Lower values indicate smoother operation"
            )
        with col10:
            oil = st.slider(
                "Oil Quality Score (0-100)",
                0.0, 100.0, 85.0,
                step=1.0,
                help="Higher values indicate better oil condition"
            )
        
        col11, col12 = st.columns(2)
        with col11:
            battery = st.slider(
                "Battery Voltage (V)",
                10.0, 16.0, 13.5,
                step=0.1,
                help="Normal range is 12-14.5V"
            )
        
        st.divider()
        
        # Submit Button
        col_submit = st.columns([1, 1, 1])
        with col_submit[1]:
            submit = st.form_submit_button(
                "⚡ Analyze Vehicle",
                use_container_width=True
            )
    
    # Process Prediction
    if submit:
        with st.spinner("🔄 Analyzing vehicle condition..."):
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
        
        st.markdown("---")
        st.markdown('<div class="section-header">📋 Diagnosis Results</div>', unsafe_allow_html=True)
        
        # Create three columns for metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.markdown(f"""
            <div class="metric-card">
                <b>Risk Score</b><br>
                <span style="font-size: 1.8em; color: #0066cc; font-weight: bold;">{prob*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown(f"""
            <div class="metric-card">
                <b>Vehicle Type</b><br>
                <span style="font-size: 1.3em; color: #0066cc;">{v_type}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col3:
            st.markdown(f"""
            <div class="metric-card">
                <b>Status</b><br>
                <span style="font-size: 1.3em; color: #0066cc;">
                {'🟢 Safe' if prob < 0.5 else '🟠 Monitor' if prob < 0.75 else '🔴 Critical'}
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Main Result Card
        if prob >= 0.75:
            result_class = "result-maintenance"
            status_icon = "🚨"
            status_text = "IMMEDIATE MAINTENANCE REQUIRED"
            advice = "Schedule maintenance immediately to prevent vehicle breakdown"
        elif prob >= 0.5:
            result_class = "result-warning"
            status_icon = "⚠️"
            status_text = "MAINTENANCE RECOMMENDED"
            advice = "Schedule maintenance within the next 1-2 weeks"
        else:
            result_class = "result-safe"
            status_icon = "✅"
            status_text = "VEHICLE IN GOOD CONDITION"
            advice = "Continue regular maintenance schedule"
        
        st.markdown(f"""
        <div class="{result_class}">
            <div class="result-title">{status_icon} {status_text}</div>
            <div class="result-subtitle">Risk Factor: {prob*100:.1f}%</div>
            <div style="margin-top: 15px; font-size: 1.1em;">{advice}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed Analysis
        st.markdown('<div class="section-header">📊 Detailed Analysis</div>', unsafe_allow_html=True)
        
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            st.markdown("### 🔧 System Components")
            st.info(f"""
            **Engine Temperature:** {temp}°C {'🔥' if temp > 100 else '✅'}
            
            **Tire Pressure:** {tire} PSI {'✅' if 30 <= tire <= 40 else '⚠️'}
            
            **Battery Voltage:** {battery}V {'✅' if 12 <= battery <= 14.5 else '⚠️'}
            """)
        
        with analysis_col2:
            st.markdown("### 📈 Performance Indicators")
            st.info(f"""
            **Oil Quality:** {oil}/100 {'✅' if oil >= 80 else '⚠️'}
            
            **Vibration Level:** {vibration} mm/s {'✅' if vibration < 3 else '⚠️'}
            
            **Brake Condition:** {brake} {'✅' if brake == 'Good' else '⚠️'}
            """)
        
        # Gauge Chart
        st.markdown("### Risk Level Gauge")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob*100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Maintenance Risk %"},
            delta={'reference': 50, 'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "#90EE90"},
                    {'range': [50, 75], 'color': "#FFD700"},
                    {'range': [75, 100], 'color': "#FF6B6B"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="white",
            font=dict(size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("---")
        st.markdown('<div class="section-header">💡 Recommendations</div>', unsafe_allow_html=True)
        
        recommendations = []
        
        if prob >= 0.5:
            recommendations.append("🔴 Schedule maintenance appointment immediately")
        if temp > 100:
            recommendations.append("❄️ Check cooling system and radiator")
        if oil < 70:
            recommendations.append("🛢️ Engine oil change required")
        if vibration > 3:
            recommendations.append("⚙️ Check alignment and suspension")
        if not (30 <= tire <= 40):
            recommendations.append("🛞 Adjust tire pressure to optimal range")
        if battery < 12 or battery > 14.5:
            recommendations.append("🔋 Battery requires inspection or replacement")
        if brake == "Poor":
            recommendations.append("🛑 Brake system urgent inspection needed")
        
        if recommendations:
            for rec in recommendations:
                st.success(rec)
        else:
            st.success("✅ All systems operating normally. Continue regular maintenance.")

elif tab_selection == "ℹ️ Information":
    st.markdown('<div class="section-header">📚 System Information</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### About Fleet Maintenance Pro
    
    **Fleet Maintenance Pro** is an advanced AI-powered predictive maintenance system designed to help you:
    
    ✅ **Predict maintenance needs** before they become critical issues
    ✅ **Reduce vehicle downtime** through proactive maintenance scheduling
    ✅ **Optimize maintenance costs** by identifying issues early
    ✅ **Improve fleet safety** with real-time vehicle health monitoring
    
    ---
    
    ### How It Works
    
    Our machine learning model analyzes multiple vehicle parameters including:
    - Engine temperature and performance
    - Tire pressure and condition
    - Oil quality and battery voltage
    - Brake condition and vibration levels
    - Historical usage and failure patterns
    
    The system provides a **risk score from 0-100%** indicating the likelihood of maintenance being needed.
    
    ---
    
    ### Risk Categories
    
    🟢 **0-50%:** Vehicle is in good condition
    🟠 **50-75%:** Maintenance recommended within 1-2 weeks
    🔴 **75-100%:** Immediate maintenance required
    """)

else:  # Quick Tips
    st.markdown('<div class="section-header">⚡ Quick Maintenance Tips</div>', unsafe_allow_html=True)
    
    tips = {
        "🌡️ Engine Temperature": "Keep your engine temperature between 85-95°C. Higher temperatures indicate cooling system issues.",
        "🛞 Tire Pressure": "Maintain tire pressure between 30-40 PSI. Check pressure monthly for optimal fuel efficiency.",
        "🛢️ Oil Quality": "Engine oil quality should be above 80/100. Change oil every 5,000-7,500 miles.",
        "🔋 Battery Voltage": "Battery voltage should be 12-14.5V. Lower voltage indicates a weak or failing battery.",
        "⚙️ Vibration Level": "Vibration below 3 mm/s is normal. Higher levels suggest alignment or bearing issues.",
        "🛑 Brake System": "Good brake condition is essential for safety. Have brakes inspected every 20,000 miles.",
        "📊 Regular Checks": "Perform maintenance checks monthly to catch issues early and extend vehicle life.",
        "🔧 Professional Service": "Always use qualified technicians for major maintenance and repairs."
    }
    
    for tip_title, tip_content in tips.items():
        st.markdown(f"""
        <div class="info-box">
            <b>{tip_title}</b><br>
            {tip_content}
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9em; margin-top: 30px;">
    <p>🚗 Fleet Maintenance Pro v1.0 | AI-Powered Vehicle Health Monitoring</p>
    <p>Last Updated: 26 February 2026</p>
</div>
""", unsafe_allow_html=True)