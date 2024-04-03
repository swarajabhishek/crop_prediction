import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('Crop_recommendation.csv')

X = df.drop('label', axis=1)
y = df['label']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, _, y_train, _ = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)


def crop_recommendation(N, P, K, temperature, humidity, ph, rainfall):
    input_data = scaler.transform([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = rf.predict(input_data)[0]
    return f"The suitable crop for this soil condition would be '{prediction}'"


st.title("AGROSENSE : SMART FARMING & CROP MANAGEMENT SYSTEM WITH IOT , ML & Data Base INTEGRATION")

N_input = st.number_input("N - Nitrogen content ratio")
P_input = st.number_input("P - Phosphorous content ratio")
K_input = st.number_input("K - Potassium content ratio")
temperature_input = st.number_input("Temperature (°C)")
humidity_input = st.number_input("Relative Humidity (%)")
ph_input = st.number_input("pH Value")
rainfall_input = st.number_input("Rainfall (mm)")

if st.button("Get Crop Recommendation"):
    recommendation = crop_recommendation(N_input, P_input, K_input, temperature_input, humidity_input, ph_input, rainfall_input)
    st.write(recommendation)
