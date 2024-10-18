import streamlit as st
import pandas as pd
from prophet import Prophet
import pickle
import pyodbc
import numpy as np

def load_model(customer_id):
    with open(r'prophet_energy_model_7.pkl', 'rb') as f:
        customer_models = pickle.load(f)
    return customer_models.get(customer_id)

def predict_for_customer(customer_id):
    model = load_model(customer_id)
    if model is None:
        return {"message": f"Model for Customer ID {customer_id} not found."}

    SERVER = '98.70.76.239,1433'
    DATABASE = 'JBVNLMISDB'
    USER = 'mis_user'
    PASS = 'Mis!2345'
    
    cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=' +
                          SERVER + ';DATABASE=' + DATABASE + ';UID=' + USER + ';PWD=' + PASS)
    cursor = cnxn.cursor()

    new_data = pd.read_sql_query("""
        SELECT DISTINCT
            a.date, 
            a.region, 
            a.cloud, 
            a.avgtemp_c, 
            a.maxwind_kph, 
            a.totalprecip_mm, 
            a.avghumidity,  
            b.event AS Holiday_type, 
            b.record_status AS Holiday
        FROM 
            JBVNLMISDB.AIML_DATA.WEATHER_DATA_JBVNL a
        JOIN 
            JBVNLMISDB.AIML_DATA.HOLIDAY_M b ON a.date = b.EFFECTIVE_DATE
        WHERE 
            a.date BETWEEN '2024-02-05' AND '2024-09-18' 
            AND a.name = 'Ranchi';
    """, cnxn)

    new_data['date'] = pd.to_datetime(new_data['date'])
    new_data.rename(columns={'date': 'ds'}, inplace=True)
    new_data['Holiday_type'] = new_data['Holiday_type'].apply(lambda x: 1 if x != 'None' else 0)

    weather_columns = ['cloud', 'avgtemp_c', 'maxwind_kph', 'totalprecip_mm', 'avghumidity']
    for col in weather_columns:
        new_data[col] = new_data[col].fillna(new_data[col].mean())

    forecast = model.predict(new_data)
    predicted_data = forecast[['ds', 'yhat']]
    return predicted_data

def run_inference():
    st.title('Load Consumption Prediction')
    st.write("Enter the Customer ID to generate predictions:")
    
    customer_id = st.text_input('Customer ID')

    if st.button('Predict'):
        if customer_id:
            result = predict_for_customer(customer_id)
            if isinstance(result, pd.DataFrame):
                st.write("Predictions for Customer ID:", customer_id)
                st.write(result)
            else:
                st.write(result['message'])
        else:
            st.write("Please enter a valid Customer ID.")

if __name__ == "__main__":
    run_inference()
