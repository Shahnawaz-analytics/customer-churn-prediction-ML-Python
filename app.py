import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="centered"
)

# ── Load Model & Scaler ───────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load('customer_churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# ── Title ─────────────────────────────────────────────────────
st.title("📡 Customer Churn Prediction")
st.markdown("Fill in the customer details below and click **Predict** to check if they are likely to churn.")
st.divider()

# ── Input Form ────────────────────────────────────────────────
st.subheader("👤 Customer Information")

col1, col2 = st.columns(2)

with col1:
    gender         = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner        = st.selectbox("Has Partner?", ["No", "Yes"])
    dependents     = st.selectbox("Has Dependents?", ["No", "Yes"])
    tenure         = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
    phone_service  = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

with col2:
    internet_service   = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security    = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup      = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection  = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support       = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv       = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies   = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

st.divider()
st.subheader("💳 Billing Information")

col3, col4 = st.columns(2)

with col3:
    contract           = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing  = st.selectbox("Paperless Billing", ["No", "Yes"])

with col4:
    payment_method  = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
    total_charges   = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=65.0*12, step=1.0)

st.divider()

# ── Predict Button ────────────────────────────────────────────
if st.button("🔍 Predict Churn", use_container_width=True, type="primary"):

    raw = {
        'gender':            gender,
        'SeniorCitizen':     1 if senior_citizen == "Yes" else 0,
        'Partner':           partner,
        'Dependents':        dependents,
        'tenure':            tenure,
        'PhoneService':      phone_service,
        'MultipleLines':     multiple_lines,
        'InternetService':   internet_service,
        'OnlineSecurity':    online_security,
        'OnlineBackup':      online_backup,
        'DeviceProtection':  device_protection,
        'TechSupport':       tech_support,
        'StreamingTV':       streaming_tv,
        'StreamingMovies':   streaming_movies,
        'Contract':          contract,
        'PaperlessBilling':  paperless_billing,
        'PaymentMethod':     payment_method,
        'MonthlyCharges':    monthly_charges,
        'TotalCharges':      total_charges,
    }

    input_df = pd.DataFrame([raw])
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # All columns x had after encoding in notebook (excluding Churn_Yes)
    expected_cols = [
        'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
        'gender_Male',
        'Partner_Yes', 'Dependents_Yes',
        'PhoneService_Yes',
        'MultipleLines_No phone service', 'MultipleLines_Yes',
        'InternetService_Fiber optic', 'InternetService_No',
        'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
        'OnlineBackup_No internet service', 'OnlineBackup_Yes',
        'DeviceProtection_No internet service', 'DeviceProtection_Yes',
        'TechSupport_No internet service', 'TechSupport_Yes',
        'StreamingTV_No internet service', 'StreamingTV_Yes',
        'StreamingMovies_No internet service', 'StreamingMovies_Yes',
        'Contract_One year', 'Contract_Two year',
        'PaperlessBilling_Yes',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed check',
    ]

    for col in expected_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[expected_cols]

    prediction  = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ **High Risk: This customer is likely to CHURN**")
        st.metric(label="Churn Probability", value=f"{probability * 100:.1f}%")
        st.markdown("""
**Suggested Actions:**
- 🎁 Offer a discount or loyalty reward
- 📞 Proactively reach out to this customer
- 📋 Suggest upgrading to a longer-term contract
        """)
    else:
        st.success("✅ **Low Risk: This customer is likely to STAY**")
        st.metric(label="Churn Probability", value=f"{probability * 100:.1f}%")
        st.markdown("This customer appears satisfied. Keep monitoring their usage patterns.")

    st.progress(float(probability))
    st.caption(f"Model confidence: {probability * 100:.1f}% chance of churn")
