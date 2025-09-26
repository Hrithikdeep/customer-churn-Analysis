
from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
import joblib, json, pickle, os, pandas as pd

# ---------------- Paths ---------------- #
import os

# robust BASE_DIR: prefer script path, fall back to cwd (works in prod and notebooks)
try:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    BASE_DIR = os.path.abspath(os.getcwd())


# Model paths (pick first existing)
MODEL_PATHS = [
    os.path.join(BASE_DIR, "models", "churn_model.pkl"),
    os.path.join(BASE_DIR, "models", "churn_model_rf.pkl"),
    os.path.join(BASE_DIR, "models", "churn_model_xgb.pkl")
]
model_path = next((p for p in MODEL_PATHS if os.path.exists(p)), None)
if model_path is None:
    raise FileNotFoundError("Model not found in models/")
model = joblib.load(model_path)

# ---------------- Feature columns ---------------- #
FEATURE_COLS_JSON = os.path.join(BASE_DIR, "models", "feature_cols.json")
FEATURE_COLS_PKL  = os.path.join(BASE_DIR, "models", "feature_cols.pkl")

if os.path.exists(FEATURE_COLS_JSON):
    with open(FEATURE_COLS_JSON, "r") as f:
        FEATURE_COLS = json.load(f)
elif os.path.exists(FEATURE_COLS_PKL):
    with open(FEATURE_COLS_PKL, "rb") as f:
        FEATURE_COLS = pickle.load(f)
else:
    raise FileNotFoundError("No feature columns file found (JSON or PKL)")

# ---------------- Prepare input ---------------- #
def prepare_input(df_input, feature_cols):
    df = df_input.copy()
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    X = pd.get_dummies(df, drop_first=True)
    # Reindex to match model features
    X = X.reindex(columns=feature_cols, fill_value=0)
    return X

# ---------------- Flask app ---------------- #
app = Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "No JSON body provided"}), 400
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return jsonify({"error": "JSON must be object or list"}), 400

        X = prepare_input(df, feature_cols=FEATURE_COLS)
        preds = model.predict(X).tolist()
        probs = model.predict_proba(X)[:,1].tolist()

        return jsonify({"predictions": preds, "probabilities": probs})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- Run Flask ---------------- #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # use 8000 by default to avoid conflicts
    # use_reloader=False prevents "Address already in use" in notebooks
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
