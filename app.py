# app.py

''' import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Load the model, vectorizer, and label encoder
model = joblib.load("condition_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Set page config
st.set_page_config(page_title="Drug Review Condition Classifier", page_icon="💊", layout="centered")

# --- HEADER ---
st.markdown("""
    <style>
        .main-title {
            font-size: 36px;
            font-weight: bold;
            color: #1F4E79;
            text-align: center;
            margin-bottom: 0;
        }
        .subtitle {
            font-size: 18px;
            color: #555;
            text-align: center;
            margin-top: 0;
        }
        .footer {
            font-size: 13px;
            color: #aaa;
            text-align: center;
            margin-top: 30px;
        }
        .stButton button {
            background-color: #1F4E79;
            color: white;
            font-size: 16px;
            padding: 8px 24px;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">💊 Patient Condition Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyze a drug review to predict the health condition: Depression, High Blood Pressure, or Type 2 Diabetes</p>', unsafe_allow_html=True)

# --- USER INPUT ---
st.markdown("### 📝 Enter a drug review below:")
user_input = st.text_area("", placeholder="Type or paste a patient review...")

# --- PREDICTION ---
if st.button("🔍 Predict Condition"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a valid review.")
    else:
        # Transform and predict
        user_vector = vectorizer.transform([user_input])
        prediction = model.predict(user_vector)
        predicted_condition = label_encoder.inverse_transform(prediction)[0]

        # --- OUTPUT CARD ---
        st.markdown("---")
        st.markdown(f"""
        <div style="padding: 20px; border: 2px solid #1F4E79; border-radius: 12px; background-color: #f0f8ff;">
            <h3 style="color: #1F4E79;">🧠 Predicted Condition:</h3>
            <h2 style="color: #007acc;">{predicted_condition}</h2>
        </div>
        """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown('<p class="footer">Made with ❤️ using Streamlit | Project: Drug Review NLP Classifier</p>', unsafe_allow_html=True)
'''


import streamlit as st
import pandas as pd

# Load your filtered dataset
# Ensure this file is the one you cleaned and used for model training
df = pd.read_csv("filtered_drug_reviews.csv")  # Replace with your cleaned dataset path

# Set Streamlit page configuration
st.set_page_config(page_title="Drug Finder", page_icon="💊", layout="centered")

# --- HEADER STYLES ---
st.markdown("""
    <style>
        .main-title {
            font-size: 36px;
            font-weight: bold;
            color: #1F4E79;
            text-align: center;
            margin-bottom: 0;
        }
        .subtitle {
            font-size: 18px;
            color: #555;
            text-align: center;
            margin-top: 0;
        }
        .footer {
            font-size: 13px;
            color: #aaa;
            text-align: center;
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# Title & Description
st.markdown('<p class="main-title">💊 Drug Info Explorer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Search by drug name to view related conditions and patient reviews</p>', unsafe_allow_html=True)

# --- USER INPUT ---
st.markdown("### 🔍 Enter a drug name:")
drug_input = st.text_input("", placeholder="e.g., Zoloft, Lisinopril, Metformin")

# --- SEARCH RESULTS ---
if st.button("Find Reviews"):
    if drug_input.strip() == "":
        st.warning("⚠️ Please enter a valid drug name.")
    else:
        # Filter the dataframe
        matches = df[df['drugName'].str.lower() == drug_input.strip().lower()]

        if matches.empty:
            st.error("❌ No data found for this drug. Please check the spelling or try another.")
        else:
            st.success(f"✅ Found {len(matches)} review(s) for **{drug_input.title()}**")

            for i, row in matches.iterrows():
                st.markdown("---")
                st.markdown(f"**🧪 Condition:** {row['condition']}")
                st.markdown(f"**⭐ Rating:** {row['rating']} / 10")
                st.markdown(f"**💬 Review:** _{row['review']}_")

# --- FOOTER ---
st.markdown('<p class="footer">Made with ❤️ using Streamlit | Drug Review Explorer</p>', unsafe_allow_html=True)
