import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

# ----------------- LOAD MODEL & SCALER -----------------
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, 'xgb_credit_model.joblib')
scaler_path = os.path.join(BASE_DIR, 'scaler.joblib')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# ----------------- APP TITLE -----------------
st.title("🛡️ Alternative Credit Scoring Engine")
st.markdown("Predict financial risk using behavioral and demographic data.")

# ----------------- INPUT COLUMNS -----------------
col1, col2 = st.columns(2)

with col1:
    st.header("Demographics")
    age = st.number_input("Age (A19)", min_value=18, max_value=100, value=30)
    
    # Gender simplified
    gender = st.selectbox("Gender", ["Female", "Male"])
    gender_value = 1.0 if gender == "Female" else 0.0

    # Education dynamic mapping
    education_levels = ["Primary", "Secondary", "Tertiary", "None/Other"]
    edu_choice = st.selectbox("Education Level", education_levels)

with col2:
    st.header("Financial Behavior")
    income = st.number_input("Estimated Monthly Income (KES)", min_value=0, value=5000)
    b3i_log = np.log1p(income)

    # Income group auto-derived
    if income < 5000:
        income_gp = 1
    elif income < 10000:
        income_gp = 2
    elif income < 20000:
        income_gp = 3
    elif income < 40000:
        income_gp = 4
    elif income < 60000:
        income_gp = 5
    elif income < 80000:
        income_gp = 6
    elif income < 100000:
        income_gp = 7
    elif income < 150000:
        income_gp = 8
    elif income < 200000:
        income_gp = 9
    else:
        income_gp = 10

    st.subheader("Usage Patterns")
    mobile_money = st.checkbox("Uses Mobile Money regularly?")
    mobile_bank = st.checkbox("Uses Mobile Banking?")
    chama = st.checkbox("Member of a Chama (Informal Group)?")
    
    # Survey flags with meaningful names
    b3a_2 = st.checkbox("Has Formal Salary?", value=False)
    b3a_3 = st.checkbox("Relies on Casual/Informal Work?", value=False)

# ----------------- PREPROCESS INPUT -----------------
def preprocess_input():
    # Education one-hot encoding
    edu_map = {
        "Primary": [1, 0, 0, 0],
        "Secondary": [0, 1, 0, 0],
        "Tertiary": [0, 0, 1, 0],
        "None/Other": [0, 0, 0, 1]
    }
    
    input_data = {
        'A19': age,
        'gender_2.0': gender_value,
        'education_2.0': edu_map[edu_choice][0],
        'education_3.0': edu_map[edu_choice][1],
        'education_4.0': edu_map[edu_choice][2],
        'education_5.0': edu_map[edu_choice][3],
        # Income group one-hot
        'incomegpnew_2.0': 1 if income_gp == 2 else 0,
        'incomegpnew_3.0': 1 if income_gp == 3 else 0,
        'incomegpnew_4.0': 1 if income_gp == 4 else 0,
        'incomegpnew_5.0': 1 if income_gp == 5 else 0,
        'incomegpnew_6.0': 1 if income_gp == 6 else 0,
        'incomegpnew_7.0': 1 if income_gp == 7 else 0,
        'incomegpnew_8.0': 1 if income_gp == 8 else 0,
        'incomegpnew_9.0': 1 if income_gp == 9 else 0,
        'incomegpnew_10.0': 1 if income_gp == 10 else 0,
        # Survey flags
        'B3A__2_1.0': 1 if b3a_2 else 0,
        'B3A__3_1.0': 1 if b3a_3 else 0,
        # Multi-level usage features
        'mobile_money_usage_2.0': 1 if mobile_money else 0,
        'mobile_money_usage_3.0': 0,
        'mobile_bank_usage_2.0': 1 if mobile_bank else 0,
        'mobile_bank_usage_3.0': 0,
        'infgp_usage_2.0': 1 if chama else 0,
        'infgp_usage_3.0': 0,
        'savings_usage_2.0': 0,
        'savings_usage_3.0': 0,
        'B3I_log': b3i_log
    }
    
    df_input = pd.DataFrame([input_data])

    # ----------------- FEATURE ORDER SAFETY -----------------
    training_order = ['A19', 'gender_2.0', 'education_2.0', 'education_3.0', 'education_4.0',
                      'education_5.0', 'incomegpnew_2.0', 'incomegpnew_3.0', 'incomegpnew_4.0',
                      'incomegpnew_5.0', 'incomegpnew_6.0', 'incomegpnew_7.0', 'incomegpnew_8.0',
                      'incomegpnew_9.0', 'incomegpnew_10.0', 'B3A__2_1.0', 'B3A__3_1.0',
                      'mobile_money_usage_2.0', 'mobile_money_usage_3.0', 'mobile_bank_usage_2.0',
                      'mobile_bank_usage_3.0', 'infgp_usage_2.0', 'infgp_usage_3.0',
                      'savings_usage_2.0', 'savings_usage_3.0', 'B3I_log']

    df_input = df_input[training_order]

    return df_input

# ----------------- PREDICTION & SHAP ------------------
if st.button("Predict Risk"):
    # 1. Preprocess input
    features_df = preprocess_input()

    # 2. Scale features
    scaled_features = scaler.transform(features_df)

    # 3. Predict
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0][1]

    # 4. High-risk threshold
    threshold = 0.4
    if probability >= threshold:
        st.error(f"High Risk Detected (Probability: {probability:.2%})")
    else:
        st.success(f"Low Risk (Probability: {probability:.2%})")

    # 5. Feature contribution (must be here, after features_df exists)
    st.subheader("Approximate Feature Contribution to Prediction")
    feature_importances = model.feature_importances_
    contrib = abs(features_df.iloc[0] * feature_importances)

    fig, ax = plt.subplots(figsize=(8, 4))
    contrib.sort_values(ascending=True).plot(kind='barh', ax=ax, color='skyblue')
    ax.set_title("Approximate Feature Contribution")
    st.pyplot(fig)

    
st.info("Note: This model uses the 2021 FinAccess Household Survey data (KNBS) to estimate risk.")