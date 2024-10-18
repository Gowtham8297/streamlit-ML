import pandas as pd
from prophet import Prophet
import pickle
import mlflow
import streamlit as st

def train_models():
    st.title("Customer Energy Consumption Model Training")

    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['date'] = pd.to_datetime(df['date'])

        customer_models = {}
        customer_ids = df['customer_id'].unique()

        mlflow.set_experiment("Prophet_Customer_Model_Training")

        if st.button("Start Training"):
            with st.spinner("Training models..."):
                for customer_id in customer_ids:
                    st.write(f"Training model for Customer ID: {customer_id}")
                    customer_data = df[df['customer_id'] == customer_id].copy()
                    customer_data.rename(columns={'date': 'ds', 'daily_consumption': 'y'}, inplace=True)

                    weather_columns = ['cloud', 'avgtemp_c', 'maxwind_kph', 'totalprecip_mm', 'avghumidity']
                    for col in weather_columns:
                        customer_data[col] = customer_data[col].fillna(customer_data[col].mean())
                    customer_data['Holiday_type'] = customer_data['Holiday_type'].apply(lambda x: 1 if x != 'None' else 0)

                    model = Prophet()
                    for col in weather_columns:
                        model.add_regressor(col)
                    model.add_regressor('Holiday_type')

                    with mlflow.start_run(run_name=f"Customer_{customer_id}_Run"):
                        model.fit(customer_data)
                        mlflow.prophet.log_model(model, f"model_{customer_id}")
                        mlflow.log_param("customer_id", customer_id)
                        training_score = model.history['y'].mean()
                        mlflow.log_metric("training_mean_consumption", training_score)

                    customer_models[customer_id] = model

                model_file = "prophet_energy_model_7.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump(customer_models, f)

                st.success(f"All customer models trained and saved to {model_file}")

if __name__ == "__main__":
    train_models()
