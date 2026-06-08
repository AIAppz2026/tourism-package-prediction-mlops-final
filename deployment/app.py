import streamlit as st
import pandas as pd
import joblib
import json
from huggingface_hub import hf_hub_download

MODEL_REPO_ID = "AIApps2026/tourism-package-model-final"

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="🌍",
    layout="centered"
)

@st.cache_resource
def load_model_and_metadata():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename="best_tourism_model.joblib",
        repo_type="model"
    )

    metadata_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename="model_metadata.json",
        repo_type="model"
    )

    model = joblib.load(model_path)

    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    return model, metadata

model, metadata = load_model_and_metadata()

st.title("🌍 Tourism Package Purchase Prediction")
st.write(
    "This app predicts whether a customer is likely to purchase the Wellness Tourism Package."
)

st.sidebar.header("Customer Input Details")

Age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35)
TypeofContact = st.sidebar.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.sidebar.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.sidebar.number_input("Duration of Pitch", min_value=0.0, max_value=150.0, value=15.0)
Occupation = st.sidebar.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.sidebar.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=3)
NumberOfFollowups = st.sidebar.number_input("Number of Followups", min_value=0.0, max_value=10.0, value=3.0)
ProductPitched = st.sidebar.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
PreferredPropertyStar = st.sidebar.number_input("Preferred Property Star", min_value=1.0, max_value=7.0, value=3.0)
MaritalStatus = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfTrips = st.sidebar.number_input("Number of Trips", min_value=0.0, max_value=30.0, value=2.0)
Passport = st.sidebar.selectbox("Passport", [0, 1])
PitchSatisfactionScore = st.sidebar.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
OwnCar = st.sidebar.selectbox("Own Car", [0, 1])
NumberOfChildrenVisiting = st.sidebar.number_input("Number of Children Visiting", min_value=0.0, max_value=5.0, value=1.0)
Designation = st.sidebar.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.sidebar.number_input("Monthly Income", min_value=0.0, max_value=200000.0, value=25000.0)

input_data = pd.DataFrame({
    "Age": [Age],
    "TypeofContact": [TypeofContact],
    "CityTier": [CityTier],
    "DurationOfPitch": [DurationOfPitch],
    "Occupation": [Occupation],
    "Gender": [Gender],
    "NumberOfPersonVisiting": [NumberOfPersonVisiting],
    "NumberOfFollowups": [NumberOfFollowups],
    "ProductPitched": [ProductPitched],
    "PreferredPropertyStar": [PreferredPropertyStar],
    "MaritalStatus": [MaritalStatus],
    "NumberOfTrips": [NumberOfTrips],
    "Passport": [Passport],
    "PitchSatisfactionScore": [PitchSatisfactionScore],
    "OwnCar": [OwnCar],
    "NumberOfChildrenVisiting": [NumberOfChildrenVisiting],
    "Designation": [Designation],
    "MonthlyIncome": [MonthlyIncome]
})

st.subheader("Input Data")
st.dataframe(input_data)

if st.button("Predict Purchase Likelihood"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("The customer is likely to purchase the tourism package.")
    else:
        st.warning("The customer is not likely to purchase the tourism package.")

    st.metric("Purchase Probability", f"{probability:.2%}")

    st.write("Model used:", metadata["best_model_name"])