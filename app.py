from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, json, pickle, os
import pandas as pd

# ---------------- Init Flask ---------------- #
app = Flask(__name__)
CORS(app)  # allow cross-origin requests (useful for Streamlit frontend etc.)

# ---------------- Paths ---------------- #
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

MODEL_PATHS = [
    os.path.join(BASE_DIR, "models", "churn_model.pkl"),
    os.path.join(BASE_DIR, "models", "churn_model_rf.pkl"),
    os.path.join(BASE_DIR, "models", "churn_model_xgb.pkl")
]

model_path = next((p for p in MODEL_PATHS if os.path.exists(p)), None)
if model_path is None:
    raise FileNotFoundError("❌ Model not found in models/ folder")

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
    raise FileNotFoundError("❌ No feature columns file found (JSON or PKL)")

# ---------------- Helpers ---------------- #
def prepare_input(df_input, feature_cols):
    df = df_input.copy()
    # Clean strings
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    # One-hot encode
    X = pd.get_dummies(df, drop_first=True)
    # Reindex to match model features
    X = X.reindex(columns=feature_cols, fill_value=0)
    return X

# ---------------- Routes ---------------- #
@app.route("/", methods=["GET"])
def home():
    return jsonify({"msg": "✅ Customer Churn API is running!"})

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "❌ No JSON body provided"}), 400

        # Single vs batch
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return jsonify({"error": "❌ JSON must be dict or list"}), 400

        X = prepare_input(df, feature_cols=FEATURE_COLS)
        preds = model.predict(X).tolist()
        probs = model.predict_proba(X)[:, 1].tolist()

        return jsonify({
            "predictions": preds,
            "probabilities": probs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- Run ---------------- #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # 8000 by default
    app.run(host="0.0.0.0", port=port, debug=True)
