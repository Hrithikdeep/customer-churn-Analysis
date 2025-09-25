import os
import joblib
import pickle
import json
import pandas as pd
import requests
import streamlit as st

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Churn Predictor Demo", layout="wide")
BASE_DIR = os.path.abspath(".")
API_URL = os.environ.get("API_URL")  # e.g., "https://your-flask-api.onrender.com"

MODEL_PATHS = [
    os.path.join(BASE_DIR, "models", "churn_model.pkl"),
    os.path.join(BASE_DIR, "models", "churn_model_rf.pkl"),
    os.path.join(BASE_DIR, "models", "churn_model_xgb.pkl")
]

FEATURE_COLS_JSON = os.path.join(BASE_DIR, "models", "feature_cols.json")
FEATURE_COLS_PKL  = os.path.join(BASE_DIR, "models", "feature_cols.pkl")

# ----------------------------
# Load local model & feature columns
# ----------------------------
def find_model():
    for p in MODEL_PATHS:
        if os.path.exists(p):
            return p
    return None

model = None
FEATURE_COLS = None
if API_URL is None:
    model_file = find_model()
    if model_file is None:
        st.error("No local model found. Set API_URL for remote predictions.")
    else:
        model = joblib.load(model_file)
        # Load feature columns
        if os.path.exists(FEATURE_COLS_JSON):
            with open(FEATURE_COLS_JSON, "r") as f:
                FEATURE_COLS = json.load(f)
        elif os.path.exists(FEATURE_COLS_PKL):
            with open(FEATURE_COLS_PKL, "rb") as f:
                FEATURE_COLS = pickle.load(f)
        else:
            st.error("No feature columns file found (.json or .pkl)")

# ----------------------------
# Local input preparation
# ----------------------------
def prepare_input_local(df):
    df2 = df.copy()
    for c in df2.select_dtypes(include="object").columns:
        df2[c] = df2[c].astype(str).str.strip()
    X = pd.get_dummies(df2, drop_first=True)
    if FEATURE_COLS is not None:
        X = X.reindex(columns=FEATURE_COLS, fill_value=0)
    return X

# ----------------------------
# App UI
# ----------------------------
st.title("Churn Predictor — Demo")
st.markdown("Upload a CSV (same columns as training) or use the sidebar single-record form.")

# ----------------------------
# Batch prediction
# ----------------------------
uploaded = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    if API_URL:
        try:
            payload = df.to_dict(orient="records")
            resp = requests.post(API_URL.rstrip("/") + "/predict", json=payload, timeout=30)
            resp.raise_for_status()
            res = resp.json()
            df["predicted_churn"] = res["predictions"]
            df["churn_probability"] = res["probabilities"]
            st.success("Predictions fetched from API")
            st.dataframe(df.head(100))
            st.download_button(
                "Download predictions CSV",
                df.to_csv(index=False).encode('utf-8'),
                "predictions.csv"
            )
        except Exception as e:
            st.error(f"API request failed: {e}")
    else:
        if model is not None:
            X = prepare_input_local(df.drop(columns=["churn"], errors="ignore"))
            df["predicted_churn"] = model.predict(X)
            df["churn_probability"] = model.predict_proba(X)[:,1]
            st.success("Local predictions ready")
            st.dataframe(df.head(100))
            st.download_button(
                "Download predictions CSV",
                df.to_csv(index=False).encode('utf-8'),
                "predictions.csv"
            )
        else:
            st.error("Local model not loaded. Cannot predict.")

# ----------------------------
# Single-record prediction
# ----------------------------
st.sidebar.header("Single-record Prediction")
# Replace these with actual feature columns
tenure = st.sidebar.number_input("tenure", min_value=0, max_value=200, value=12)
monthlycharges = st.sidebar.number_input("monthlycharges", min_value=0.0, value=70.0)
contract = st.sidebar.selectbox("contract", ["Month-to-month", "One year", "Two year"])
paymentmethod = st.sidebar.selectbox("paymentmethod", ["Electronic check", "Mailed check", 
                                                        "Bank transfer (automatic)", "Credit card (automatic)"])

if st.sidebar.button("Predict single"):
    payload = {
        "tenure": tenure,
        "monthlycharges": monthlycharges,
        "contract": contract,
        "paymentmethod": paymentmethod
    }
    if API_URL:
        try:
            resp = requests.post(API_URL.rstrip("/") + "/predict", json=payload, timeout=15)
            resp.raise_for_status()
            r = resp.json()
            prob = r["probabilities"][0]
            st.metric("Churn probability", f"{prob:.2%}")
        except Exception as e:
            st.error(f"API request failed: {e}")
    else:
        if model is not None:
            df_single = pd.DataFrame([payload])
            Xs = prepare_input_local(df_single)
            prob = model.predict_proba(Xs)[:,1][0]
            st.metric("Churn probability", f"{prob:.2%}")
        else:
            st.error("Local model not loaded. Cannot predict.")
