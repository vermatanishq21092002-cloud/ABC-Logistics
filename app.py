import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="Delivery Delay Predictor",
    page_icon="🚚",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("model.sav")

model = load_model()

st.title("🚚 Delivery Delay Predictor")
st.write("Enter the values below to predict whether the delivery will be delayed.")

st.divider()

# The model expects 11 input variables in this order.
feature_names = [feature_names = ['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
       'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
       'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
       'Warehouse_Processing_Time']

    "Feature 1",
    "Feature 2",
    "Feature 3",
    "Feature 4",
    "Feature 5",
    "Feature 6",
    "Feature 7",
    "Feature 8",
    "Feature 9",
    "Feature 10",
    "Feature 11"
]

inputs = []

col1, col2 = st.columns(2)

for i, feature in enumerate(feature_names):
    column = col1 if i % 2 == 0 else col2

    with column:
        value = st.number_input(
            feature,
            value=0.0,
            step=1.0
        )
        inputs.append(value)

st.divider()

if st.button("Predict Delivery Delay", type="primary", use_container_width=True):

    input_data = np.array(inputs).reshape(1, -1)

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.error("⚠️ Prediction: DELIVERY DELAY")
    else:
        st.success("✅ Prediction: NO DELIVERY DELAY")

    st.subheader("Prediction Probability")

    classes = model.classes_

    for class_value, probability in zip(classes, probabilities):

        if class_value == 1:
            label = "Delivery Delay"
        else:
            label = "No Delivery Delay"

        st.write(f"**{label}: {probability:.2%}**")
        st.progress(float(probability))

st.caption("Model: Logistic Regression")
