import streamlit as st
# @st.cache_resource(show_spinner="🔍 Loading ML models...")
def load_model():
    import joblib
    sale_model = joblib.load("models/sale_model.pkl")
    # stock_model = joblib.load("models/stock_model.pkl")
    stock_model = joblib.load("models/xgboost_20250709_111210.pkl")
    oos_model = joblib.load("models/LightGBM_severity_model.pkl")
    return sale_model, stock_model, oos_model
