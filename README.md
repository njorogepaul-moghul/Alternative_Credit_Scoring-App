# 🛡️ Alternative Credit Scoring Engine
***[😊 TRY Alternative Credit Scoring app here:](https://alternativecreditscoring-app-gojkhneajqbfpnshgxsyv3.streamlit.app/)***


Predict financial risk using behavioral and demographic data from the 2021 FinAccess Household Survey (KNBS). Built with Streamlit and XGBoost.

---

## Features

- User-friendly Streamlit interface with two input columns:
  - **Demographics:** Age, Gender, Education Level  
  - **Financial Behavior:** Monthly Income, Income Group, Mobile Money/Bank usage, Chama participation  
- Predicts **High vs Low Financial Risk** with probability  
- **Feature contribution bar chart** shows approximate impact of each input on prediction  
- Adjustable **risk threshold** (default: 0.4)  
- Survey flags included (B3A__2: Employment Income, B3A__3: Casual Work Income)  

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo

Create a virtual environment:

python -m venv venv
# Activate:
# Windows
source venv/Scripts/activate
# macOS/Linux
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt
Usage
streamlit run Alternative_app.py

Input your data in the provided fields.

Click Predict Risk.

View the risk probability and feature contribution chart.

Model Information

Model: XGBoost Classifier (xgb_credit_model.joblib)

Scaler: StandardScaler (scaler.joblib)

Features: 26 input features including demographics, financial behavior, and usage patterns

Threshold: 0.4 probability for high risk

Notes

Ensure venv/ is ignored in Git (included in .gitignore)

The feature contribution chart uses approximate feature importance for Streamlit compatibility (SHAP optional)

App tested with Python 3.12 and Streamlit >=1.25

Requirements

Python 3.12+

streamlit

pandas

numpy

joblib

matplotlib

xgboost

(Optional for SHAP visualizations: shap, ipython)

Contact

Paul Njoroge

Email: larneymogul@gmail.com
