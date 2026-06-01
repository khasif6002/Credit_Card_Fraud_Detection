import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("fraud_detection_rf.pkl")

# App title
st.title("Credit Card Fraud Detection System")

st.write("Upload a CSV file to detect fraudulent transactions")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    # Read CSV
    data = pd.read_csv(uploaded_file)

    st.write("Uploaded Dataset")
    st.dataframe(data.head(), use_container_width = True)

    # Prediction
    predictions = model.predict(data)

    # Add predictions column
    data['Prediction'] = predictions

    # Count fraud transactions
    fraud_count = (predictions == 1).sum()

    # Display results
    st.write("Prediction Results")
    st.dataframe(data.head(), use_container_width = True)

    st.write(f"Total Fraud Transactions Detected: {fraud_count}")

    # Show only fraud transactions
    fraud_transactions = data[data['Prediction'] == 1]

    st.write("Fraudulent Transactions")
    st.dataframe(fraud_transactions)

    # Download results
    csv = data.to_csv(index=False)

    st.download_button(
        label="Download Results CSV",
        data=csv,
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )