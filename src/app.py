import streamlit as st
import joblib
import numpy as np
import re

# Load the trained model
model = joblib.load("models/breast_cancer_model.pkl")

# Internal symptom-to-feature mapping
symptom_mapping = {
    "hard lump": ("radius_mean", 30.0),
    "bloody discharge": ("texture_mean", 35.0),
    "inverted nipple": ("smoothness_mean", 0.20),
    "skin dimpling": ("compactness_mean", 0.25),
    "persistent pain": ("concavity_mean", 0.30),
    "rapid swelling": ("perimeter_mean", 200.0),
    "redness": ("fractal_dimension_mean", 0.15)
}

def map_symptoms_to_features(user_input):
    """Convert user symptoms into numerical feature values."""
    input_features = np.zeros(31)  # Model expects 31 features
    user_input = user_input.lower()
    
    matched = False  
    for symptom, (feature, value) in symptom_mapping.items():
        if re.search(symptom, user_input):
            feature_index = list(symptom_mapping.values()).index((feature, value))
            input_features[feature_index] = value
            matched = True
    
    return input_features, matched

# Set page config
st.set_page_config(page_title="PinkCode - Breast Cancer Detection", layout="centered")

# Custom CSS for full-page light pink background
st.markdown("""
    <style>
        body {
            background-color: #ffe6f2 !important;  /* Light Pink Background */
        }
        .main {
            background-color: #ffe6f2; /* Light pink container */
            padding: 20px;
            border-radius: 10px;
        }
        .stApp {
            background-color: #ffe6f2; /* Light pink background for the entire page */
        }
        .stButton>button {
            background-color: #ff4b4b; 
            color: white; 
            font-size: 18px;
            border-radius: 8px;
        }
        .stTextInput>div>div>input {
            font-size: 16px; 
            padding: 10px;
            border-radius: 5px;
        }
        .logo {
            display: flex;
            justify-content: center;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Enlarged Breast Cancer Logo
st.markdown("<div class='logo'><img src='https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Pink_ribbon.svg/800px-Pink_ribbon.svg.png' width='150'></div>", unsafe_allow_html=True)

# App Title
st.title("🎀 PinkCode - Breast Cancer Symptom Checker")
st.write("Describe your breast-related symptoms below and get an AI-based assessment.")

# User Input
user_symptoms = st.text_area("✏️ Enter your symptoms:", height=100)

if st.button("🔍 Check for Cancer Risk"):
    features, matched = map_symptoms_to_features(user_symptoms)
    
    if matched:
        prediction = model.predict([features])[0]
    else:
        prediction = 0  

    # Display Results
    st.markdown("---")
    
    if prediction == 1:
        st.error("⚠️ **Malignant (Cancer Detected).** Please visit your nearest hospital for further screening.")
        st.write("### 🚑 What to do next:")
        st.write("- Schedule an appointment with an oncologist immediately.")
        st.write("- Avoid self-diagnosis; only a doctor can confirm cancer.")
        st.write("- Consider a mammogram or biopsy for further testing.")
        st.write("- Stay positive and seek support from family and specialists.")
    else:
        st.success("✅ **Benign (No Cancer Detected).** Maintain a healthy lifestyle and regular check-ups to prevent cancer.")
        st.write("### 🛡️ Prevention Tips:")
        st.write("- 🥗 Maintain a balanced diet rich in fruits and vegetables.")
        st.write("- 🏃 Exercise regularly to reduce risk factors.")
        st.write("- 🚭 Avoid smoking and excessive alcohol consumption.")
        st.write("- 🩺 Perform monthly self-breast exams and go for regular check-ups.")

st.markdown("---")
st.write("💖 *Empowering Women with AI* | Developed by **Kris**")
