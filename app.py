import streamlit as st
import pandas as pd
import pickle
from annotated_text import annotated_text
import numpy as np

# Load the models and encoders
with open('models/rainfall_model.pkl', 'rb') as file:
    rainfall_model = pickle.load(file)

with open('models/label_encoder.pkl', 'rb') as file:
    rainfall_label_encoder = pickle.load(file)

with open('models/tmean_model.pkl', 'rb') as file:
    tmean_model = pickle.load(file)

with open('models/label_encoder_tmean.pkl', 'rb') as file:
    tmean_label_encoder = pickle.load(file)

with open('models/rice_model.pkl', 'rb') as file:
    rice_model = pickle.load(file)

with open('models/rice_label_encoder.pkl', 'rb') as file:
    rice_label_encoder = pickle.load(file)

# Load rice data for details
rice_data = pd.read_csv('data/Rice.csv')  # Update the path if necessary
rice_details = rice_data.set_index('rice_name').to_dict(orient='index')

# Thai month names to numbers mapping
thai_months = {
    "มกราคม": 1, "กุมภาพันธ์": 2, "มีนาคม": 3, "เมษายน": 4, "พฤษภาคม": 5, 
    "มิถุนายน": 6, "กรกฎาคม": 7, "สิงหาคม": 8, "กันยายน": 9, "ตุลาคม": 10, 
    "พฤศจิกายน": 11, "ธันวาคม": 12
}

# Initialize session state variables for results if they don't exist
if 'rainfall_result' not in st.session_state:
    st.session_state.rainfall_result = None

if 'tmean_result' not in st.session_state:
    st.session_state.tmean_result = None

if 'rice_result' not in st.session_state:
    st.session_state.rice_result = None

# Streamlit app title
st.title('Talk To The Tree')

# Tabs for Rainfall, Tmean, and Rice Predictions
tab1, tab2, tab3 = st.tabs(["การทำนายปริมาณน้ำฝน", "การทำนายอุณหภูมิ", "การทำนายชนิดข้าว"])

# Tab 1: Rainfall Prediction
with tab1:
    st.header('การทำนายปริมาณน้ำฝน')

    # User inputs for rainfall prediction
    province_rainfall = st.selectbox('กรุณาเลือกจังหวัด', rainfall_label_encoder.classes_, key='rainfall_province')
    forecast_month_thai_rainfall = st.selectbox('กรุณาเลือกเดือน', list(thai_months.keys()), key='rainfall_month')
    
    if st.button('ทำนายปริมาณน้ำฝน'):
        # Convert Thai month name to month number
        forecast_month_rainfall = thai_months[forecast_month_thai_rainfall]
        
        # Encode the province
        province_encoded_rainfall = rainfall_label_encoder.transform([province_rainfall])[0]

        # Make prediction
        rainfall_prediction = rainfall_model.predict([[province_encoded_rainfall, forecast_month_rainfall]])

        # Store result in session state
        st.session_state.rainfall_result = annotated_text(
            "ปริมาณน้ำฝนในจังหวัด ", 
            (f"{province_rainfall}", "", "#8ef"), 
            " ในเดือน ", 
            (f"{forecast_month_thai_rainfall}", ""), 
            " : ", 
            (f"{rainfall_prediction[0]:.2f} มม.", "", "#faf")
        )
        
    # Display stored result
    if st.session_state.rainfall_result:
        st.write(st.session_state.rainfall_result)

# Tab 2: Tmean Prediction
with tab2:
    st.header('การทำนายอุณหภูมิ')

    # User inputs for Tmean prediction
    province_tmean = st.selectbox('กรุณาเลือกจังหวัด', tmean_label_encoder.classes_, key='tmean_province')
    forecast_month_thai_tmean = st.selectbox('กรุณาเลือกเดือน', list(thai_months.keys()), key='tmean_month')
    
    if st.button('ทำนายอุณหภูมิ'):
        # Convert Thai month name to month number
        forecast_month_tmean = thai_months[forecast_month_thai_tmean]
        
        # Encode the province
        province_encoded_tmean = tmean_label_encoder.transform([province_tmean])[0]

        # Make prediction
        try:
            tmean_prediction = tmean_model.predict([[province_encoded_tmean, forecast_month_tmean]])
            st.session_state.tmean_result = annotated_text(
                "อุณหภูมิในจังหวัด ", 
                (f"{province_tmean}", "", "#8ef"), 
                " ในช่วง ", 
                (f"{forecast_month_thai_tmean}", ""), 
                " : ", 
                (f"{tmean_prediction[0]:.2f} °C", "", "#faf")
            )
        except Exception as e:
            st.session_state.tmean_result = f"An error occurred: {e}"
    
    # Display stored result
    if st.session_state.tmean_result:
        st.write(st.session_state.tmean_result)

# Tab 3: Rice Prediction
with tab3:
    st.header('ทำนายชนิดข้าวที่ควรปลูก')

    # User inputs for rice prediction
    rain_mm = st.number_input('กรุณาเลือกปริมาณน้ำฝน (มม.)', min_value=0.0, max_value=1000.0, step=0.1, key='rice_rain')
    tmax_tmax = st.number_input('กรุณาเลือกอุณหภูมิสูงสุด (°C)', min_value=0.0, max_value=100.0, step=0.1, key='rice_tmax')

    if st.button('ทำนายชนิดข้าวที่ควรปลูก'):
        # Make prediction
        try:
            # Get prediction probabilities for each class
            probabilities = rice_model.predict_proba([[tmax_tmax, rain_mm]])
            
            # Get top 3 rice types based on probabilities
            top_indices = np.argsort(probabilities[0])[-3:][::-1]
            top_rice_names = rice_label_encoder.inverse_transform(top_indices)
            top_probabilities = probabilities[0][top_indices]

            # Fetch details for top 3 rice types
            results = []
            for i in range(3):
                rice_name = top_rice_names[i]
                prob = top_probabilities[i]
                details = rice_details.get(rice_name, {})
                
                result_text = (
                    f"อันดับที่ {i+1})\n"
                    f"ชื่อข้าว: {rice_name}\n"
                    f"ประเภทข้าว: {details.get('rice_type', 'N/A')}\n"
                    f"อายุเก็บเกี่ยว: {details.get('Harvesting_age', 'N/A')} วัน\n"
                    f"ผลผลิต: {details.get('Yield', 'N/A')} กก./ไร่\n"
                    f"คุณลักษณะเด่น: {details.get('Highlights', 'N/A')}\n"
                    f"ปริมาณน้ำฝน: {details.get('rain_mm', 'N/A'):.2f} มม."
                )
                results.append(
                    f"{result_text}"
                )

            st.session_state.rice_result = results
        except Exception as e:
            st.session_state.rice_result = f"An error occurred: {e}"
    
    # Display stored result
    if st.session_state.rice_result:
        for result in st.session_state.rice_result:
            st.write(result)
