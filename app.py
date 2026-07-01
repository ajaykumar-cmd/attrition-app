"""
app.py
------
Streamlit app for the Employee Attrition Prediction model.

Run from the terminal:
    streamlit run app.py
"""

import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="📊")

# ----------------------------------------------------------------------
# Load the trained model + metadata (created by train_model.py)
# ----------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    return joblib.load("model_bundle.pkl")

bundle = load_bundle()
model = bundle["model"]
feature_columns = bundle["feature_columns"]
numeric_cols = bundle["numeric_cols"]
categorical_cols = bundle["categorical_cols"]
category_options = bundle["category_options"]

st.title("📊 Employee Attrition Predictor")
st.write(
    "Enter an employee's details below to predict whether they are "
    "likely to leave the company."
)

# ----------------------------------------------------------------------
# Collect inputs
# ----------------------------------------------------------------------
with st.form("attrition_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=70, value=30)
        daily_rate = st.number_input("Daily Rate", min_value=0, value=800)
        distance = st.number_input("Distance From Home (km)", min_value=0, value=5)
        education = st.selectbox("Education Level (1=Below College, 5=Doctor)", [1, 2, 3, 4, 5], index=2)

    with col2:
        business_travel = st.selectbox("Business Travel", category_options["BusinessTravel"])
        department = st.selectbox("Department", category_options["Department"])
        education_field = st.selectbox("Education Field", category_options["EducationField"])

    submitted = st.form_submit_button("Predict")

# ----------------------------------------------------------------------
# Build a single-row dataframe that matches the training feature layout
# ----------------------------------------------------------------------
if submitted:
    raw_input = pd.DataFrame([{
        "Age": age,
        "DailyRate": daily_rate,
        "DistanceFromHome": distance,
        "Education": education,
        "BusinessTravel": business_travel,
        "Department": department,
        "EducationField": education_field,
    }])

    encoded_input = pd.get_dummies(raw_input, columns=categorical_cols, drop_first=True)

    # Add any missing dummy columns (categories not chosen) as 0,
    # then reorder to exactly match training column order.
    for col in feature_columns:
        if col not in encoded_input.columns:
            encoded_input[col] = 0
    encoded_input = encoded_input[feature_columns].astype(float)

    prediction = model.predict(encoded_input)[0]
    probability = model.predict_proba(encoded_input)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"⚠️ Likely to leave (Attrition risk: {probability:.1%})")
    else:
        st.success(f"✅ Likely to stay (Attrition risk: {probability:.1%})")

    with st.expander("See model input (debug)"):
        st.dataframe(encoded_input)
