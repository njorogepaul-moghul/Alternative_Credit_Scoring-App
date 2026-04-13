# 🛡️ Alternative Credit Scoring Engine — Streamlit App
### Interactive Deployment of an XGBoost Credit Risk Model for Kenya's Informal Economy

[![Live App](https://img.shields.io/badge/🚀_Live_App-Streamlit-FF4B4B)](https://alternativecreditscoring-app-gojkhneajqbfpnshgxsyv3.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Data](https://img.shields.io/badge/Data-KNBS_FinAccess_2021-green)

---

## 📌 Overview

This is the **deployed application** for the Alternative Credit Scoring System — an end-to-end machine learning project that predicts household loan default risk using **behavioral and demographic data** from Kenya's 2021 FinAccess Household Survey (KNBS/CBK).

The app makes the underlying XGBoost model accessible to anyone — no technical knowledge required. A user inputs demographic and financial behavior data and instantly receives a **High Risk / Low Risk prediction** with a probability score and feature contribution breakdown.

**Underlying model:** Tuned XGBoost · ROC-AUC: 0.6681 · High-Risk Recall: 68%

---

## 🚀 Live Demo

👉 [**Try the App Here**](https://alternativecreditscoring-app-gojkhneajqbfpnshgxsyv3.streamlit.app/)

---

## 🎯 What Makes This App Different

Most credit scoring apps assume bank statements and formal income records. This one doesn't. It scores individuals using **alternative data signals** specifically validated for Kenya's informal economy:

| Signal Type | Features Used |
|---|---|
| Demographics | Age, Gender, Education Level |
| Financial Behavior | Monthly Income (KES), Income Group (auto-derived) |
| Digital Footprint | Mobile Money usage, Mobile Banking usage |
| Social Capital | Chama (Informal Group) membership |
| Employment Type | Formal Salary vs Casual/Informal Work |

---

## ⚙️ How It Works

### 1. Input Collection
Two-column layout separates **Demographics** from **Financial Behavior** for a clean UX:
- Dropdowns and checkboxes for categorical inputs
- Number inputs for age and income (KES)
- Income group (1–10) is **auto-derived** from raw income — user never sees the encoding

### 2. Preprocessing Pipeline
The app replicates the exact training pipeline:
```python
# Income log-transformation (matches training)
b3i_log = np.log1p(income)

# One-hot encoding of education, income group, usage patterns
# Feature order locked to training order (26 features)
# StandardScaler applied before prediction
scaled_features = scaler.transform(features_df)
```

### 3. Prediction & Threshold
```python
probability = model.predict_proba(scaled_features)[0][1]
threshold = 0.4  # Tuned for higher recall on high-risk class

if probability >= threshold:
    → High Risk (probability displayed as %)
else:
    → Low Risk (probability displayed as %)
```

> The 0.4 threshold (vs standard 0.5) is deliberate — in credit risk, catching more defaulters (higher recall) is prioritised over minimising false positives.

### 4. Feature Contribution Chart
After each prediction, an **approximate feature contribution bar chart** is rendered showing which inputs drove the risk score — making the model's decision interpretable to non-technical users.

---

## 🛠️ Tech Stack

```
Python · XGBoost · Streamlit · Pandas · NumPy · Joblib · Matplotlib · SHAP
```

---

## 🖥️ Run Locally

```bash
git clone https://github.com/njorogepaul-moghul/Alternative_Credit_Scoring-App.git
cd Alternative_Credit_Scoring-App
pip install -r requirements.txt
streamlit run Alternative_app.py
```

---

## 📁 Repository Structure

```
├── Alternative_app.py        # Streamlit app — UI, preprocessing, prediction
├── xgb_credit_model.joblib   # Trained XGBoost model
├── scaler.joblib             # StandardScaler — required for input preprocessing
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

---

## ⚠️ Important Note on scaler.joblib

The `scaler.joblib` file **must be present** in the same directory as `Alternative_app.py` for the app to run. It applies the same StandardScaler fitted during training — without it, predictions will be incorrect.

---

## 🔗 Related Repository

Full model training pipeline, EDA, and phase-by-phase documentation:
👉 [Alternative Credit Scoring System — Full Project](https://github.com/njorogepaul-moghul/Alternative-credit-scoring)

---

## 📊 Model Performance (Underlying XGBoost)

| Metric | Value |
|---|---|
| ROC-AUC | 0.6681 |
| High-Risk Recall | 0.68 |
| Low-Risk Precision | 0.77 |
| Overall Accuracy | 0.61 |
| Risk Threshold | 0.40 |

---

## 📬 Contact

**Paul Njoroge** | larneymogul@gmail.com | Kenyatta University, Kenya

> Data source: 2021 FinAccess Household Survey — Kenya National Bureau of Statistics (KNBS) & Central Bank of Kenya (CBK)
