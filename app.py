import streamlit as st
import pandas as pd
import numpy as np
import joblib

MODEL_PATH = "svm_heart_model.pkl"  # your trained model file

st.set_page_config(page_title="Heart Failure Prediction", layout="centered")
st.title("❤️ Heart Failure Prediction using SVM")

@st.cache_resource
def load_model():
    model_bundle = joblib.load(MODEL_PATH)
    return model_bundle["model"], model_bundle["feature_cols"]

try:
    model, feature_cols = load_model()
except Exception as e:
    st.error(f"Model file not found. Upload svm_heart_model.pkl. Error: {e}")
    st.stop()

st.sidebar.header("Choose Input Method")
mode = st.sidebar.radio("Select Input Method:", ["Manual Input", "Upload CSV"])

# ---------------------------- MANUAL INPUT ----------------------------------
if mode == "Manual Input":
    st.subheader("Enter Patient Details")

    inputs = {}
    for col in feature_cols:
        inputs[col] = st.number_input(col, min_value=0.0, value=0.0)

    if st.button("Predict"):
        X = np.array([list(inputs.values())])
        pred = model.predict(X)[0]
        st.success(f"Predicted Outcome: **{pred}**")

# ---------------------------- CSV UPLOAD ------------------------------------
else:
    st.subheader("Upload CSV File for Batch Predictions")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            preds = model.predict(df[feature_cols])
            df["Predicted"] = preds
            st.dataframe(df)

            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions", csv_data, "predictions.csv")
