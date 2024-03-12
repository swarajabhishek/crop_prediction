import pandas as pd
import gradio as gr
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

    input_data = scaler.transform(
        [[N, P, K, temperature, humidity, ph, rainfall]])

    prediction = rf.predict(input_data)[0]

    return f"The suitable crop for this soil condition would be '{prediction}'"


N_input = gr.Number(label="N - Nitrogen content ratio")
P_input = gr.Number(label="P - Phosphorous content ratio")
K_input = gr.Number(label="K - Potassium content ratio")
temperature_input = gr.Number(label="Temperature (°C)")
humidity_input = gr.Number(label="Relative Humidity (%)")
ph_input = gr.Number(label="pH Value")
rainfall_input = gr.Number(label="Rainfall (mm)")


output = gr.Textbox(label="Crop Recommendation")

iface = gr.Interface(fn=crop_recommendation,
                     inputs=[N_input, P_input, K_input, temperature_input, humidity_input,
                             ph_input, rainfall_input],
                     outputs=output,
                     description="<h1 style='text-align:center;font-weight:bold;'>AGROSENSE : SMART FARMING & CROP MANAGEMENT SYSTEM WITH IOT , ML & DB INTEGRATION</h1>",
                     title="19ELC381 - OPEN LAB")

iface.launch(share=True)
