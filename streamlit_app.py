# streamlit_app.py
import os
import joblib
import pickle
import json
import pandas as pd
import requests
import streamlit as st

# optional plotting libs
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    PLOTTING_OK = True
except Exception:
    PLOTTING_OK = False

# ----------------------------
# Config (use file's directory so paths work no matter where you run it from)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="Customer Churn Analysis", layout="wide")
API_URL = os.environ.get("API_URL")  # optional: remote Flask API

# Paths (use the exact cleaned path you gave)
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned", "Telco-Customer-Churn.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATHS = [
    os.path.join(MODEL_DIR, "churn_model.pkl"),
    os.path.join(MODEL_DIR, "churn_model_rf.pkl"),
    os.path.join(MODEL_DIR, "churn_model_xgb.pkl"),
    os.path.join(MODEL_DIR, "churn_model_pipeline.pkl"),
    os.path.join(MODEL_DIR, "xgb_model.pkl"),
]
FEATURE_COLS_JSON = os.path.join(MODEL_DIR, "feature_cols.json")
FEATURE_COLS_PKL  = os.path.join(MODEL_DIR, "feature_cols.pkl")

# ----------------------------
# Helper: find and load model + feature columns
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
    if model_file:
        try:
            model = joblib.load(model_file)
        except Exception as e:
            st.warning(f"Found model at {model_file} but failed to load: {e}")
            model = None
        # load feature columns (json preferred)
        if os.path.exists(FEATURE_COLS_JSON):
            with open(FEATURE_COLS_JSON, "r") as f:
                FEATURE_COLS = json.load(f)
        elif os.path.exists(FEATURE_COLS_PKL):
            with open(FEATURE_COLS_PKL, "rb") as f:
                FEATURE_COLS = pickle.load(f)
    else:
        # no local model found
        pass

# ----------------------------
# Input preparation (safe)
# ----------------------------
def prepare_input_local(df):
    df2 = df.copy()
    # trim whitespace in object cols
    for c in df2.select_dtypes(include="object").columns:
        df2[c] = df2[c].astype(str).str.strip()
    X = pd.get_dummies(df2, drop_first=True)
    if FEATURE_COLS:
        # align with training features (fills missing with 0)
        X = X.reindex(columns=FEATURE_COLS, fill_value=0)
    return X

# ----------------------------
# Sidebar navigation & debug
# ----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["EDA Insights", "Prediction", "Debug"])

# small debug info about paths
if st.sidebar.checkbox("Show paths (debug)"):
    st.sidebar.write("App base dir:", BASE_DIR)
    st.sidebar.write("Looking for cleaned CSV at:", DATA_PATH)
    st.sidebar.write("Model loaded:", bool(model))
    st.sidebar.write("Feature cols loaded:", bool(FEATURE_COLS))

# ----------------------------
# EDA Insights Page
# ----------------------------
if page == "EDA Insights":
    st.title("📊 Customer Churn — Insights")
    st.markdown("This view shows quick EDA charts from the cleaned dataset.")

    st.write("Looking for cleaned file at:", f"`{DATA_PATH}`")
    if not os.path.exists(DATA_PATH):
        st.error("Cleaned dataset not found at the above path. Please place your cleaned CSV there.")
    else:
        df = pd.read_csv(DATA_PATH)
        # normalize column names (strip)
        df.columns = [c.strip() for c in df.columns]

        # Some datasets use 'Churn' or 'churn' — handle both
        churn_col = None
        for c in df.columns:
            if c.lower() == "churn":
                churn_col = c
                break

        # Safe fallback: show head
        st.subheader("Data preview")
        st.dataframe(df.head(5))

        # Plot 1: Churn distribution
        st.subheader("Churn distribution")
        if churn_col is not None:
            if PLOTTING_OK:
                fig, ax = plt.subplots(figsize=(6,3))
                sns.countplot(x=churn_col, data=df, ax=ax, palette="Set2")
                ax.set_title("Churn Distribution")
                st.pyplot(fig)
            else:
                st.write(df[churn_col].value_counts())
        else:
            st.info("No `Churn` column found in dataset to plot distribution.")

        # Plot 2: Monthly charges vs churn
        st.subheader("Monthly Charges vs Churn")
        # attempt common column names
        mc_col = None
        for candidate in ["MonthlyCharges", "monthlycharges", "Monthly Charges", "monthly_charges"]:
            if candidate in df.columns:
                mc_col = candidate
                break
        if mc_col and churn_col:
            if PLOTTING_OK:
                fig, ax = plt.subplots(figsize=(7,4))
                sns.boxplot(x=churn_col, y=mc_col, data=df, ax=ax, palette="Set2")
                ax.set_title("Monthly Charges by Churn")
                st.pyplot(fig)
            else:
                st.write(df.groupby(churn_col)[mc_col].describe())
        else:
            st.info("MonthlyCharges or Churn column not found for this plot.")

        # Plot 3: Contract vs churn
        st.subheader("Contract Type vs Churn")
        contract_col = None
        for candidate in ["Contract", "contract"]:
            if candidate in df.columns:
                contract_col = candidate
                break
        if contract_col and churn_col:
            if PLOTTING_OK:
                fig, ax = plt.subplots(figsize=(8,4))
                sns.countplot(x=contract_col, hue=churn_col, data=df, ax=ax, palette="Set2")
                ax.set_title("Contract Type vs Churn")
                plt.xticks(rotation=20)
                st.pyplot(fig)
            else:
                st.write(pd.crosstab(df[contract_col], df[churn_col], normalize="index"))
        else:
            st.info("Contract or Churn column not found for this plot.")

        # Plot 4: Tenure distribution
        st.subheader("Tenure distribution")
        tenure_col = None
        for candidate in ["tenure", "Tenure"]:
            if candidate in df.columns:
                tenure_col = candidate
                break
        if tenure_col:
            if PLOTTING_OK:
                fig, ax = plt.subplots(figsize=(7,4))
                sns.histplot(df[tenure_col].dropna(), bins=30, kde=True)
                ax.set_title("Tenure Distribution")
                st.pyplot(fig)
            else:
                st.write(df[tenure_col].describe())
        else:
            st.info("Tenure column not found for this plot.")

# ----------------------------
# Prediction Page
# ----------------------------
elif page == "Prediction":
    st.title("🔮 Churn Prediction")
    st.markdown("Upload a CSV (same columns as training) or use the sidebar to enter a single record.")

    uploaded = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        # save a small preview
        st.subheader("Preview of uploaded data")
        st.dataframe(df.head(5))

        if API_URL:
            try:
                payload = df.to_dict(orient="records")
                resp = requests.post(API_URL.rstrip("/") + "/predict", json=payload, timeout=30)
                resp.raise_for_status()
                res = resp.json()
                df["predicted_churn"] = res["predictions"]
                df["churn_probability"] = res["probabilities"]
                st.success("Predictions fetched from remote API")
                st.dataframe(df.head(50))
                st.download_button("Download predictions CSV", df.to_csv(index=False).encode("utf-8"), "predictions.csv")
            except Exception as e:
                st.error(f"API request failed: {e}")
        else:
            if model is None:
                st.error("No local model available. Either set API_URL or place a model in models/")
            else:
                X = prepare_input_local(df.drop(columns=["Churn", "churn"], errors="ignore"))
                try:
                    df["predicted_churn"] = model.predict(X)
                    df["churn_probability"] = model.predict_proba(X)[:,1]
                    st.success("Local predictions ready")
                    st.dataframe(df.head(50))
                    st.download_button("Download predictions CSV", df.to_csv(index=False).encode("utf-8"), "predictions.csv")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    # Single-record in sidebar
    st.sidebar.header("Single-record Input")
    # Use common columns — adjust if your dataset has different names
    tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=200, value=12)
    monthlycharges = st.sidebar.number_input("Monthly Charges", min_value=0.0, value=70.0, format="%.2f")
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paymentmethod = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    if st.sidebar.button("Predict single"):
        payload = {
            "tenure": tenure,
            "MonthlyCharges": monthlycharges,
            "Contract": contract,
            "PaymentMethod": paymentmethod
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
            if model is None:
                st.error("No local model available. Place a model file in models/ or set API_URL.")
            else:
                df_single = pd.DataFrame([payload])
                Xs = prepare_input_local(df_single)
                try:
                    prob = model.predict_proba(Xs)[:,1][0]
                    st.metric("Churn probability", f"{prob:.2%}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# ----------------------------
# Debug page (optional)
# ----------------------------
elif page == "Debug":
    st.title("Debug / Environment info")
    st.write("BASE_DIR:", BASE_DIR)
    st.write("DATA_PATH:", DATA_PATH)
    st.write("Cleaned file exists:", os.path.exists(DATA_PATH))
    st.write("Model present:", find_model())
    st.write("Model loaded:", bool(model))
    st.write("Feature cols loaded:", bool(FEATURE_COLS))
    if os.path.exists(DATA_PATH):
        st.write("Cleaned file shape:", pd.read_csv(DATA_PATH).shape)

