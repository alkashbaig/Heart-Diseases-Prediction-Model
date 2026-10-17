import streamlit as st
import pandas as pd
import joblib

# Load the trained machine learning model and its supporting files
model = joblib.load("LR_heart.pkl")
scaler = joblib.load("scaler_heart.pkl")
expected_columns = joblib.load("columns_heart.pkl")

# Page title
st.title("❤️ Heart Disease Risk Prediction")
st.write("Enter the details below to estimate the likelihood of heart disease.")

st.info("This tool is for educational purposes only. It cannot diagnose a medical condition.")

# 1. Basic information
st.header("1. Basic Information")

age = st.slider("Age (in years)", 18, 100, 40)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)
sex = {"Male": "M", "Female": "F"}[sex]

# 2. Heart-related information
st.header("2. Heart-Related Information")

chest_pain = st.selectbox(
    "Type of Chest Pain",
    [
        "Pain during physical activity",
        "Chest discomfort not related to the heart",
        "Typical heart-related chest pain",
        "No chest pain"
    ]
)

chest_pain_mapping = {
    "Pain during physical activity": "ATA",
    "Chest discomfort not related to the heart": "NAP",
    "Typical heart-related chest pain": "TA",
    "No chest pain": "ASY"
}
chest_pain = chest_pain_mapping[chest_pain]

resting_bp = st.number_input(
    "Blood Pressure at Rest (mm Hg)",
    min_value=80, max_value=200, value=120,
    help="Blood pressure measured when the body is resting."
)

cholesterol = st.number_input(
    "Cholesterol Level (mg/dL)",
    min_value=100, max_value=600, value=200,
    help="A measurement of cholesterol in the blood."
)

fasting_bs = st.selectbox(
    "Is fasting blood sugar above 120 mg/dL?",
    ["No", "Yes"]
)
fasting_bs = {"No": 0, "Yes": 1}[fasting_bs]

resting_ecg = st.selectbox(
    "Heart's Electrical Activity (ECG)",
    [
        "Normal",
        "ST-T wave changes",
        "Left ventricular hypertrophy"
    ],
    help="An ECG records the electrical activity of the heart."
)

resting_ecg_mapping = {
    "Normal": "Normal",
    "ST-T wave changes": "ST",
    "Left ventricular hypertrophy": "LVH"
}
resting_ecg = resting_ecg_mapping[resting_ecg]

max_hr = st.slider(
    "Maximum Heart Rate (beats per minute)",
    60, 220, 150,
    help="The highest heart rate recorded during exercise or testing."
)

exercise_angina = st.selectbox(
    "Chest Pain During Exercise?",
    ["No", "Yes"]
)
exercise_angina = {"No": "N", "Yes": "Y"}[exercise_angina]

oldpeak = st.slider(
    "ECG Change During Exercise (Oldpeak)",
    0.0, 6.0, 1.0, step=0.1,
    help="A numerical measurement of changes in the ECG during exercise."
)

st_slope = st.selectbox(
    "ECG Pattern During Exercise",
    ["Upward", "Flat", "Downward"]
)

st_slope_mapping = {
    "Upward": "UP",
    "Flat": "Flat",
    "Downward": "Down"
}
st_slope = st_slope_mapping[st_slope]

# 3. Make a prediction
st.header("3. Prediction")

if st.button("Predict Heart Disease Risk", type="primary"):

    # Arrange the entered information for the model
    input_data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1
    }

    # Convert the information into a table
    input_df = pd.DataFrame([input_data])

    # Add missing model columns with a value of zero
    for column in expected_columns:
        if column not in input_df.columns:
            input_df[column] = 0

    # Arrange columns in the same order used during training
    input_df = input_df[expected_columns]

    # Prepare the data for the trained model
    scaled_input = scaler.transform(input_df)

    # Generate the prediction
    prediction = model.predict(scaled_input)[0]

    # Display the result
    if prediction == 1:
        st.error("⚠️ The model indicates a higher likelihood of heart disease.")
    else:
        st.success("✅ The model indicates a lower likelihood of heart disease.")

    st.caption(
        "This is a machine learning estimate, not a medical diagnosis. "
        "Consult a qualified healthcare professional for medical advice."
    )
