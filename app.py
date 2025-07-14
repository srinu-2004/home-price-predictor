import streamlit as st
import joblib
import pandas as pd


bundle = joblib.load("home_price_rf.pkl")
model = bundle["model"]
feature_names = bundle["feature_names"]


st.title("🏠 Home Price Predictor")

st.markdown("Enter the property details to predict the house price:")


user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"{feature}", value=0.0)


if st.button("Predict Price"):
    input_df = pd.DataFrame([user_input])[feature_names]
    prediction = model.predict(input_df)[0]
    price_in_inr = prediction * 100_000 * 83  # 1 USD ≈ ₹83
    st.success(f"Estimated House Price: **₹{price_in_inr:,.0f}**")

