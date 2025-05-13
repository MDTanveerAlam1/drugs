# app.py

import streamlit as st
import joblib
import numpy as np

# Load the model, vectorizer, and label encoder
model = joblib.load("condition_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# App title
st.title("Patient Condition Classifier Based on Drug Review")

# User input
user_input = st.text_area("Enter a drug review:")

if st.button("Predict Condition"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Clean and transform the input
        user_vector = vectorizer.transform([user_input])
        prediction = model.predict(user_vector)
        predicted_condition = label_encoder.inverse_transform(prediction)[0]

        # Show result
        st.success(f"Predicted Condition: **{predicted_condition}**")