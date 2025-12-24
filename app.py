import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the saved model and column names
model = joblib.load('churn_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# 2. App Title and Description
st.title("📉 Telco Customer Churn Prediction")
st.write("Enter customer details to predict if they will leave (churn) or stay.")

# 3. Create Input Form (Sidebar for better layout)
st.sidebar.header("User Input Features")

def user_input_features():
    # Numeric Inputs
    tenure = st.sidebar.slider('Tenure (Months)', 1, 72, 12)
    monthly_charges = st.sidebar.number_input('Monthly Charges ($)', min_value=18.0, max_value=120.0, value=70.0)
    total_charges = st.sidebar.number_input('Total Charges ($)', min_value=18.0, max_value=9000.0, value=1000.0)
    
    # Categorical Inputs (Dropdowns)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.sidebar.selectbox('Senior Citizen', [0, 1]) # 1=Yes, 0=No
    partner = st.sidebar.selectbox('Has Partner?', ['Yes', 'No'])
    dependents = st.sidebar.selectbox('Has Dependents?', ['Yes', 'No'])
    phone_service = st.sidebar.selectbox('Phone Service', ['Yes', 'No'])
    multiple_lines = st.sidebar.selectbox('Multiple Lines', ['No phone service', 'No', 'Yes'])
    internet_service = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.sidebar.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    online_backup = st.sidebar.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
    device_protection = st.sidebar.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
    tech_support = st.sidebar.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    streaming_tv = st.sidebar.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    streaming_movies = st.sidebar.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
    contract = st.sidebar.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.sidebar.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.sidebar.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

    # Create a dictionary of inputs
    data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method
    }
    return pd.DataFrame(data, index=[0])

# Get user input
input_df = user_input_features()

# Display input for verification
st.subheader('Customer Details')
st.write(input_df)

# 4. Predict Button
if st.button('Predict Churn'):
    # Encode the input (Turn text to numbers)
    input_encoded = pd.get_dummies(input_df)
    
    # Align columns with the trained model (fill missing cols with 0)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    # Make Prediction
    prediction = model.predict(input_encoded)
    prediction_prob = model.predict_proba(input_encoded)
    
    # 5. Show Results
    st.subheader('Prediction Result:')
    
    if prediction[0] == 1:
        st.error(f"⚠️ High Risk! This customer is likely to CHURN. (Probability: {prediction_prob[0][1]*100:.2f}%)")
    else:
        st.success(f"✅ Safe! This customer is likely to STAY. (Probability: {prediction_prob[0][0]*100:.2f}%)")