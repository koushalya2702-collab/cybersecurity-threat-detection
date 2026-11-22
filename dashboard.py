
import streamlit as st
import pandas as pd
import requests
import numpy as np

from ml_model import get_features

# -------------------------------
# PAGE SETTINGS
# -------------------------------

st.set_page_config(
    page_title="Cybersecurity Threat Detection",
    page_icon="🛡️",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"

# -------------------------------
# HEADER
# -------------------------------

st.title("🛡️ Cybersecurity Threat Detection System")

st.write(
    "Upload network traffic data and detect whether "
    "the traffic is BENIGN or an ATTACK."
)

st.divider()

# -------------------------------
# CHECK API CONNECTION
# -------------------------------

st.subheader("API Connection")

try:
    response = requests.get(API_URL, timeout=5)

    if response.status_code == 200:
        st.success("FastAPI is connected!")
    else:
        st.warning("FastAPI is running, but check its home endpoint.")

except requests.exceptions.RequestException:
    st.error("FastAPI is not running. Start your FastAPI server first.")
    st.stop()

st.divider()

# -------------------------------
# UPLOAD CSV
# -------------------------------

st.subheader("Upload Network Traffic CSV")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        df.columns = df.columns.str.strip()

        st.success("CSV file uploaded successfully!")

        st.write("Dataset Preview")
        st.dataframe(df.head())

        st.write("Dataset Shape:", df.shape)

        # -------------------------------
        # GET REQUIRED MODEL FEATURES
        # -------------------------------

        required_features = get_features()

        missing_features = [
            feature for feature in required_features
            if feature not in df.columns
        ]

        if missing_features:
            st.error(
                "Some required model features are missing from your CSV."
            )
            st.write("Missing features:")
            st.write(missing_features)
            st.stop()

        # -------------------------------
        # SELECT RECORD
        # -------------------------------

        st.divider()
        st.subheader("Select Network Traffic Record")

        row_number = st.number_input(
            "Select row number",
            min_value=0,
            max_value=len(df) - 1,
            value=0,
            step=1
        )

        selected_row = df.iloc[int(row_number)]

        st.write("Selected Record")
        st.dataframe(
            selected_row[required_features].to_frame().T
        )

        # -------------------------------
        # PREDICTION
        # -------------------------------

        if st.button("🔍 Detect Threat", type="primary"):

            features = {}

            for feature in required_features:
                value = pd.to_numeric(
                    pd.Series([selected_row[feature]]),
                    errors="coerce"
                ).iloc[0]

                if pd.isna(value):
                    st.error(
                        f"Invalid or missing value in feature: {feature}"
                    )
                    st.stop()

                features[feature] = float(value)

            payload = {
                "features": features
            }

            try:
                prediction_response = requests.post(f"{API_URL}/analyze",
                json=payload,
                timeout=30
            )

                if prediction_response.status_code == 200:

                    result = prediction_response.json()

                    st.divider()
                    st.subheader("Detection Result")

                    st.json(result)

                    prediction = str(
                        result.get("prediction", "")
                    ).upper()

                    if "BENIGN" in prediction:
                        st.success("✅ BENIGN TRAFFIC")
                    elif "ATTACK" in prediction:
                        st.error("🚨 ATTACK DETECTED")
                    else:
                        st.info(
                            "Prediction received. "
                            "Check the response above."
                        )

                else:
                    st.error(
                        f"API Error: {prediction_response.status_code}"
                    )
                    st.code(prediction_response.text)

            except requests.exceptions.RequestException as error:
                st.error(f"Could not connect to prediction API: {error}")

    except Exception as error:
        st.error("Something went wrong while reading the CSV.")
        st.exception(error)

else:
    st.info("Upload a CSV file to begin threat detection.")

st.divider()

st.caption("Cybersecurity Threat Detection System | CSE Project")
