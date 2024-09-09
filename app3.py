import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError

# Load the models and encoders
with open('models/rainfall_model.pkl', 'rb') as file:
    rainfall_model = pickle.load(file)

with open('models/label_encoder.pkl', 'rb') as file:
    rainfall_label_encoder = pickle.load(file)

with open('models/tmean_model.pkl', 'rb') as file:
    tmean_model = pickle.load(file)

with open('models/label_encoder_tmean.pkl', 'rb') as file:
    tmean_label_encoder = pickle.load(file)

# Thai month names to numbers mapping
thai_months = {
    "มกราคม": 1, "กุมภาพันธ์": 2, "มีนาคม": 3, "เมษายน": 4, "พฤษภาคม": 5, 
    "มิถุนายน": 6, "กรกฎาคม": 7, "สิงหาคม": 8, "กันยายน": 9, "ตุลาคม": 10, 
    "พฤศจิกายน": 11, "ธันวาคม": 12
}

# Initialize session state variables for results if they don't exist
if 'combined_results' not in st.session_state:
    st.session_state.combined_results = None

# Streamlit app title
st.title('Weather Prediction App')

# Single tab for combined prediction
st.header('Combined Rainfall and Tmean Prediction')

# User inputs for combined prediction
province = st.selectbox('Select Province', rainfall_label_encoder.classes_, key='province')
forecast_month_thai = st.selectbox('Select Forecast Month', list(thai_months.keys()), key='month')

if st.button('Predict'):
    # Convert Thai month name to month number
    forecast_month = thai_months[forecast_month_thai]
    
    try:
        # Encode the province
        province_encoded_rainfall = rainfall_label_encoder.transform([province])[0]
        province_encoded_tmean = tmean_label_encoder.transform([province])[0]

        # Make predictions
        rainfall_prediction = rainfall_model.predict([[province_encoded_rainfall, forecast_month]])
        tmean_prediction = tmean_model.predict([[province_encoded_tmean, forecast_month]])
        
        # Store results in session state
        st.session_state.combined_results = {
            'rainfall': f"Predicted Rainfall for {province} in {forecast_month_thai}: {rainfall_prediction[0]:.2f} mm",
            'tmean': f"Predicted Mean Temperature (Tmean) for {province} in {forecast_month_thai}: {tmean_prediction[0]:.2f} °C"
        }

    except ValueError as e:
        if "unseen labels" in str(e):
            st.error("The selected province is not recognized. Please ensure it matches the training data.")
        else:
            st.error(f"An error occurred: {e}")

# Display stored results
if st.session_state.combined_results:
    st.write(st.session_state.combined_results['rainfall'])
    st.write(st.session_state.combined_results['tmean'])
