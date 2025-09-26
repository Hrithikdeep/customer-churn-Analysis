# 📊 Customer Churn Analysis — End-to-End Project

## 🚀 Project Overview
This project demonstrates a **full Data Science & Machine Learning workflow** for predicting customer churn (e.g., Telecom/Banking).  
It includes:
- Data Cleaning & Preprocessing  
- Exploratory Data Analysis (EDA)  
- Machine Learning Models (Logistic Regression, Random Forest, XGBoost)  
- Model Deployment via **Flask API**  
- Interactive Dashboard via **Streamlit App**

The goal: help businesses **identify customers likely to churn** and take proactive actions.

---

## 📂 Project Structure

```text
Customer-Churn-Analysis/
│
├── data/
│   ├── raw/                  # Raw datasets (e.g., Telco-Customer-Churn.csv)
│   ├── cleaned/              # Cleaned datasets ready for analysis
│   └── analysis/             # EDA outputs, charts, summaries, and predictions_for_tableau.csv
│       ├── charts/           # PNG/SVG screenshots of EDA plots
│       ├── summaries/        # CSV/JSON summary tables (e.g., aggregations)
│       └── predictions_for_tableau.csv
│
├── notebooks/                # Jupyter Notebooks (Phase 1–3)
│   ├── 01_data_preparation.ipynb
│   ├── 02_eda.ipynb
│   └── 03_modeling.ipynb
│
├── models/                   # Saved models + feature columns
│   ├── churn_model_xgb.pkl           # XGBoost final model (example)
│   ├── churn_model_rf.pkl            # RandomForest model (optional)
│   ├── churn_model_pipeline.pkl      # Pipeline (preprocessor + estimator)
│   ├── churn_model.pkl               # generic model file (if used)
│   ├── feature_cols.json             # final encoded feature column order (used for inference)
│   └── feature_cols_encoded.json     # optional: list of post-encoding column names
│
├── app.py                    # Flask API (Phase 4) — serves /ping and /predict
├── streamlit_app.py          # Streamlit dashboard (EDA + Prediction)
├── requirements.txt          # Python dependencies
├── Procfile                  # For Render deployment (Flask)
└── README.md                 # This documentation


---

```

## 🏗️ Phases Breakdown

### **Phase 1 — Data Preparation**
- Load raw dataset (`Telco-Customer-Churn.csv`)
- Handle missing values
- Encode categorical variables
- Save cleaned dataset → `data/cleaned/Telco-Customer-Churn.csv`



---

### **Phase 2 — Exploratory Data Analysis (EDA)**
- Visualized churn distribution
- Relationship between **tenure, monthly charges, contract type** and churn
- Correlation heatmaps

 
---

### **Phase 3 — Modeling**
- Tried multiple models:
  - Logistic Regression
  - Random Forest
  - XGBoost (best performing)
- Hyperparameter tuning (GridSearchCV)
- Evaluated using Accuracy, Precision, Recall, F1, ROC-AUC
- Saved final model + features:
  - `models/churn_model_xgb.pkl`
  - `models/feature_cols.json`
    

---

### **Phase 4 — Deployment**

#### **Flask API**
- `app.py` exposes:
  - `/ping` → health check  
  - `/predict` → returns churn predictions + probabilities

**Run locally:**
```bash
pip install -r requirements.txt
python app.py

```

 API runs at → http://localhost:8000

## Sample Request (curl):
 
 
  curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"tenure": 12, "MonthlyCharges": 70, "Contract": "Month-to-month", "PaymentMethod": "Electronic check"}'

 ## Response:
  
  {
  "predictions": [1],
  "probabilities": [0.82]
}

## Streamlit App

streamlit_app.py provides:

EDA Insights tab (churn charts, contract analysis, tenure distribution)

Prediction tab (single + batch prediction)

Run locally:
streamlit run streamlit_app.py

## 🌐 Deployment
Flask API → deployed on Render

Live:  https://customer-churn-analysis-1-o034.onrender.com/

Streamlit App → deployed on Streamlit Cloud

Live: https://customer-churn-analysis-6sxepmjiqkuyauabkgngme.streamlit.app/ 

## ⚡ Key Skills Demonstrated: 

Data Cleaning & Preprocessing (pandas, sklearn)

Exploratory Data Analysis (matplotlib, seaborn)

Machine Learning (Logistic Regression, Random Forest, XGBoost)

Model Saving & Loading (joblib, pickle)

API Development (Flask, gunicorn, flask-cors)

Dashboard Development (Streamlit)

Cloud Deployment (Render + Streamlit Cloud)

---

## How to Reproduce

``

1- Clone repo:

            git clone <repo-url>
            cd Customer-Churn-Analysis

 2- Install requirements:  

             pip install -r requirements.txt

3- Run notebooks step by step for Phase 1–3

4-Start Flask API or Streamlit app for Phase 4

---

## Future Improvements
Add AutoML for better model selection

Connect to live database (PostgreSQL / MySQL)

Integrate alerts for high churn risk customers

Add authentication for API

## Demo Screenshots

1- EDA Insights tab

  <img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/86a4fc63-11dd-4651-abdd-d1dbf783f0f8" />
  
  <img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/96561dfa-4184-48a2-a78d-cd43ea2dda26" />

  <img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/8b398ed9-8be9-4968-87aa-d2cb20677503" />

  <img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/d6930470-b541-4876-9ed4-d23d7cbbdbd6" />

---

## 🌐 Live Demo

 [![Streamlit Dashboard](https://img.shields.io/badge/Streamlit-Dashboard-orange?style=for-the-badge&logo=streamlit)](https://customer-churn-analysis-6sxepmjiqkuyauabkgngme.streamlit.app/)

 [![Flask API](https://img.shields.io/badge/Flask-API-blue?style=for-the-badge&logo=flask)](https://customer-churn-analysis-1-o034.onrender.com/)

 ---
 
 ## 👨‍💻 About the Author

# Hrithik Deep
📍 Aspiring Data Analyst / Data Scientist

---

## 🌐 Connect with me

[![GitHub](https://img.shields.io/badge/GitHub-Hrithikdeep-black?style=for-the-badge&logo=github)](https://github.com/Hrithikdeep)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-YourLinkedIn-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/hrithikdeep/)  
[![Email](https://img.shields.io/badge/Email-hrithikdeep.ds@gmail.com-red?style=for-the-badge&logo=gmail)](hrithikdeep.ds@gmail.com)

---

## ⭐ Acknowledgments

Dataset: Telco Customer Churn Dataset (Kaggle)

Libraries: pandas, seaborn, scikit-learn, XGBoost, Flask, Streamlit


     









