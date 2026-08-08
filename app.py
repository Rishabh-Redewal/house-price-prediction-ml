import streamlit as st
import pandas as pd
import joblib

# Load trained model and preprocessing pipeline
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠"
)

st.title("🏠 House Price Prediction")
st.write("Enter the house details below to predict its median house value.")

# Input fields

longitude = st.number_input(
    "Longitude",
    value=-122.23,
    format="%.4f"
)

latitude = st.number_input(
    "Latitude",
    value=37.88,
    format="%.4f"
)

housing_median_age = st.number_input(
    "Housing Median Age",
    min_value=1.0,
    value=30.0
)

total_rooms = st.number_input(
    "Total Rooms",
    min_value=1.0,
    value=2000.0
)

total_bedrooms = st.number_input(
    "Total Bedrooms",
    min_value=1.0,
    value=400.0
)

population = st.number_input(
    "Population",
    min_value=1.0,
    value=1000.0
)

households = st.number_input(
    "Households",
    min_value=1.0,
    value=300.0
)

median_income = st.number_input(
    "Median Income",
    min_value=0.0,
    value=4.0
)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    [
        "<1H OCEAN",
        "INLAND",
        "NEAR OCEAN",
        "NEAR BAY",
        "ISLAND"
    ]
)

# Prediction button
if st.button("🔮 Predict House Price"):

    input_data = pd.DataFrame({
        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income],
        "ocean_proximity": [ocean_proximity]
    })

    # Transform input using saved pipeline
    transformed_input = pipeline.transform(input_data)

    # Make prediction
    prediction = model.predict(transformed_input)

    st.success(
        f"🏠 Predicted House Value: ${prediction[0]:,.2f}"
    )