
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="centered"
)

# --------------------------------------------------
# 2. Load saved artifacts
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    base_dir = Path(__file__).resolve().parent

    model = joblib.load(base_dir / "LR_heart.pkl")
    scaler = joblib.load(base_dir / "scaler.pkl")
    columns = joblib.load(base_dir / "columns.pkl")

    return model, scaler, columns


try:
    model, scaler, expected_columns = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the model files: {exc}")
    st.stop()

# Exact feature schema from your training data
EXPECTED_COLUMNS = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "MaxHR",
    "Oldpeak",
    "Sex_M",
    "ChestPainType_ATA",
    "ChestPainType_NAP",
    "ChestPainType_TA",
    "RestingECG_Normal",
    "RestingECG_ST",
    "ExerciseAngina_Y",
    "ST_Slope_Flat",
    "ST_Slope_Up"
]

NUM_COLS = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "MaxHR",
    "Oldpeak"
]

# Validate saved feature list
if list(expected_columns) != EXPECTED_COLUMNS:
    st.error(
        "The columns.pkl file does not match the expected training schema. "
        "Please check the saved columns and model artifacts."
    )
    st.stop()

if hasattr(scaler, "feature_names_in_"):
    if list(scaler.feature_names_in_) != NUM_COLS:
        st.error(
            "The scaler was not fitted on the expected five numerical "
            "features. Please verify your training code."
        )
        st.stop()

if hasattr(model, "n_features_in_"):
    if model.n_features_in_ != len(EXPECTED_COLUMNS):
        st.error(
            "The saved model does not expect 15 input features. "
            "Your model and columns file may come from different training runs."
        )
        st.stop()

# --------------------------------------------------
# 3. App introduction
# --------------------------------------------------
st.title("❤️ Heart Disease Risk Prediction")

st.write(
    "Enter the health measurements below to obtain a prediction "
    "from the trained machine-learning model."
)

st.info(
    "Educational project only. This application cannot diagnose "
    "heart disease or replace medical evaluation."
)

# --------------------------------------------------
# 4. Collect input
# --------------------------------------------------
st.header("1. Basic Information")

age = st.slider("Age (years)", 18, 100, 40)

sex_label = st.selectbox("Sex", ["Male", "Female"])
sex = {"Male": "M", "Female": "F"}[sex_label]

st.header("2. Heart-Related Information")

chest_pain_label = st.selectbox(
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
chest_pain = chest_pain_mapping[chest_pain_label]

resting_bp = st.number_input(
    "Blood Pressure at Rest (mm Hg)",
    min_value=80,
    max_value=200,
    value=120
)

cholesterol = st.number_input(
    "Cholesterol (mg/dL)",
    min_value=100,
    max_value=600,
    value=200
)

fasting_bs_label = st.selectbox(
    "Fasting blood sugar above 120 mg/dL?",
    ["No", "Yes"]
)
fasting_bs = {"No": 0, "Yes": 1}[fasting_bs_label]

resting_ecg_label = st.selectbox(
    "Resting ECG",
    [
        "Normal",
        "ST-T wave changes",
        "Left ventricular hypertrophy"
    ]
)

resting_ecg_mapping = {
    "Normal": "Normal",
    "ST-T wave changes": "ST",
    "Left ventricular hypertrophy": "LVH"
}
resting_ecg = resting_ecg_mapping[resting_ecg_label]

max_hr = st.slider(
    "Maximum Heart Rate (beats per minute)",
    60, 220, 150
)

exercise_angina_label = st.selectbox(
    "Chest Pain During Exercise?",
    ["No", "Yes"]
)
exercise_angina = {"No": "N", "Yes": "Y"}[exercise_angina_label]

oldpeak = st.slider(
    "Oldpeak (ECG change during exercise)",
    min_value=0.0,
    max_value=6.0,
    value=1.0,
    step=0.1
)

st_slope_label = st.selectbox(
    "ST Slope",
    ["Upward", "Flat", "Downward"]
)

st_slope_mapping = {
    "Upward": "Up",
    "Flat": "Flat",
    "Downward": "Down"
}
st_slope = st_slope_mapping[st_slope_label]

# --------------------------------------------------
# 5. Build model input
# --------------------------------------------------
st.header("3. Prediction")

if st.button("Predict Heart Disease Risk", type="primary"):

    # Numerical features
    input_data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,

        # One-hot encoded categories.
        # Reference categories are represented by zeros.
        "Sex_M": int(sex == "M"),

        "ChestPainType_ATA": int(chest_pain == "ATA"),
        "ChestPainType_NAP": int(chest_pain == "NAP"),
        "ChestPainType_TA": int(chest_pain == "TA"),

        "RestingECG_Normal": int(resting_ecg == "Normal"),
        "RestingECG_ST": int(resting_ecg == "ST"),

        "ExerciseAngina_Y": int(exercise_angina == "Y"),

        "ST_Slope_Flat": int(st_slope == "Flat"),
        "ST_Slope_Up": int(st_slope == "Up")
    }

    # Preserve the exact training order and use floating-point
    # values so scaled decimals can be assigned safely.
    input_df = pd.DataFrame(
        [[input_data[col] for col in EXPECTED_COLUMNS]],
        columns=EXPECTED_COLUMNS,
        dtype=float
    )

    try:
        # Scale numerical columns only.
        # Do not refit the scaler during prediction.
        input_df[NUM_COLS] = scaler.transform(input_df[NUM_COLS])

        # Predict using the saved model
        prediction = model.predict(input_df)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(
                "⚠️ The model predicts the positive heart disease class."
            )
        else:
            st.success(
                "✅ The model predicts the negative heart disease class."
            )

        st.caption(
            "This is an educational machine-learning prediction, "
            "not a medical diagnosis. Consult a qualified healthcare "
            "professional for medical advice."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.warning(
            "Check that the model, scaler, and columns were saved "
            "from the same training run and use the same preprocessing."
        )
