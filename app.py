import streamlit as st
import pandas as pd
import numpy as np
import joblib

MODEL_PATH = "svm_heart_model.pkl"   # your saved model file

st.set_page_config(page_title="Heart Failure Prediction", layout="centered")

st.title("❤️ Heart Failure Prediction using SVM")
st.write("Enter patient features or upload CSV to predict heart failure risk.")

# Load model
@st.cache_resource
def load_model():
    try:
        model_bundle = joblib.load(MODEL_PATH)
        return model_bundle["model"], model_bundle["feature_cols"], model_bundle["label_encoder"]
    except:
        st.error("Model file not found! Upload svm_heart_model.pkl to your GitHub repo.")
        st.stop()

model, feature_cols, label_encoder = load_model()

# Sidebar
st.sidebar.header("Choose Input Mode")
mode = st.sidebar.radio("Select:", ["Manual Input", "Upload CSV File"])

# --------------------------------------------------------
# MANUAL INPUT MODE
# --------------------------------------------------------
if mode == "Manual Input":
    st.subheader("Enter Patient Data")

    inputs = []
    for col in feature_cols:
        value = st.number_input(f"{col}", min_value=0.0, max_value=300.0, value=1.0)
        inputs.append(value)

    if st.button("Predict"):
        X = np.array(inputs).reshape(1, -1)
        pred = model.predict(X)[0]
        pred_label = label_encoder.inverse_transform([pred])[0]

        st.success(f"Prediction: **{pred_label}**")

# --------------------------------------------------------
# CSV UPLOAD MODE
# --------------------------------------------------------
else:
    st.subheader("Upload CSV File for Multiple Predictions")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            preds = model.predict(df[feature_cols])
            preds_labels = label_encoder.inverse_transform(preds)
            df["Prediction"] = preds_labels

            st.write(df.head())
            st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv")
