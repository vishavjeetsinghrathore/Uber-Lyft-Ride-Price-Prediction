import streamlit as st
import pandas as pd
import joblib

st.title("🚕 Uber / Lyft Cab Price Predictor")

distance = st.number_input("Distance (km)", 0.1, 100.0)
surge = st.slider("Surge Multiplier", 1.0, 3.0, 1.0)

cab_type = st.selectbox("Cab Type", ["Uber", "Lyft"])
source = st.text_input("Source")
destination = st.text_input("Destination")

if st.button("Predict Price"):
    input_df = pd.DataFrame({
        "distance": [distance],
        "surge_multiplier": [surge],
        "cab_type": [cab_type],
        "source": [source],
        "destination": [destination]
    })

    model = joblib.load("best_model.pkl")
    price = model.predict(input_df)

    st.success(f"Estimated Price: ₹{price[0]:.2f}")
