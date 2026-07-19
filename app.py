import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# --- Load the trained model artifacts ---
with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

st.title("Customer Churn Predictor")
st.write("Upload a CSV of customers (same format as the Telco Churn dataset) to get churn risk scores.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.subheader("Preview of uploaded data")
    st.dataframe(raw_df.head())

    df = raw_df.copy()

    # --- Same cleaning steps as our notebook ---
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(0)

    if 'SeniorCitizen' in df.columns and df['SeniorCitizen'].dtype != 'object':
        df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

    cols_to_fix = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].replace({'No phone service': 'No', 'No internet service': 'No'})

    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies',
                   'PaperlessBilling', 'SeniorCitizen']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    multi_cols = ['gender', 'InternetService', 'Contract', 'PaymentMethod']
    multi_cols_present = [c for c in multi_cols if c in df.columns]
    df = pd.get_dummies(df, columns=multi_cols_present, drop_first=True)

    # --- Align columns to exactly match what the model was trained on ---
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0  # a category never seen in this upload gets a 0

    customer_ids = df['customerID'] if 'customerID' in df.columns else pd.Series(range(len(df)))
    df_model_ready = df[model_columns]

    # --- Scale numeric columns using the SAME scaler fit during training ---
    numeric_cols = [c for c in ['tenure', 'MonthlyCharges'] if c in df_model_ready.columns]
    df_scaled = df_model_ready.copy()
    df_scaled[numeric_cols] = scaler.transform(df_model_ready[numeric_cols])

    # --- Predict ---
    churn_probs = model.predict_proba(df_scaled)[:, 1]

    threshold = st.slider(
        "Churn risk threshold (customers above this score are flagged as at-risk)",
        min_value=0.0, max_value=1.0, value=0.70, step=0.01
    )

    results = pd.DataFrame({
        'customerID': customer_ids,
        'churn_probability': churn_probs.round(3),
        'flagged_at_risk': (churn_probs >= threshold)
    }).sort_values('churn_probability', ascending=False)

    st.subheader("Churn risk scores")
    st.dataframe(results, use_container_width=True)

    st.metric("Customers flagged at-risk", int(results['flagged_at_risk'].sum()))

    csv_out = results.to_csv(index=False).encode('utf-8')
    st.download_button("Download results as CSV", csv_out, "churn_predictions.csv", "text/csv")
