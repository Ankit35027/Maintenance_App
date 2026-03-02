# ML-Based Vehicle Maintenance Prediction

This project implements a classical machine learning workflow to predict vehicle maintenance requirements using historical usage logs and sensor telemetry. By shifting the maintenance strategy from reactive to predictive, the system aims to minimize unexpected breakdowns and high operational costs.

## 📊 Dataset Overview
The model is trained on a dataset containing **40,000 vehicle records**.
* **Numerical Features:** Usage Hours, Engine Temperature (Celsius), Tire Pressure, Oil Quality, Battery Voltage, Vibration Level, and Maintenance Cost.
* **Categorical Features:** Vehicle Type (Car, Truck, Bus) and Brake Condition (Good, Fair, Poor).
* **Target Variable:** Maintenance Required (Binary: '1' for high risk, '0' for safe).

## ⚙️ Methodology
* **Preprocessing:** Standardized anomalous temperature readings, handled missing values via median/mode imputation (SimpleImputer), and applied `StandardScaler` for normalization.
* **EDA Highlights:** Analysis revealed that **Engine Temperature** has the highest correlation (0.52) with failure risk, while **Oil Quality** shows a strong negative correlation (-0.35).
* **Algorithms:** Evaluated **Logistic Regression** as a robust baseline and **Decision Tree** classifiers to capture non-linear relationships.
* **Optimization:** Conducted hyperparameter tuning for the Decision Tree, identifying an optimal `max_depth` of 11 to prevent overfitting.

## 📈 Performance Results
Evaluated on an 8,000-sample test split:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **86.84%** | **86.37%** | **83.36%** | **84.84%** |
| Decision Tree (Tuned) | 82.89% | 82.73% | 77.42% | 79.99% |



> **Key Finding:** Logistic Regression is the recommended architecture because it achieved the highest **Recall** (83.36%), which is critical for operational safety to avoid missing actual breakdowns.