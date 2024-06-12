import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from datetime import date 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
model_1 = load_model("LSTM_Stock Prediction Model.keras")
model_2 = load_model("GRU_Stock_Prediction_Model.keras")
model_3 = load_model("TCN_Stock_Prediction_Model.keras")

# HTML and CSS to change the font color
st.markdown(
    """
    <style>
    .title {
        color: #ADD8E6;
        font-size: 36px;
    }
    .header {
        color: #90EE90;
        font-size: 30px;
    }
    .subheader {
        color: #FFB6C1;
        font-size: 24px;
    }
    .text {
        color: #D3D3D3;
        font-size: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="title">Stock Market Predictor</h1>', unsafe_allow_html=True)
stock = st.text_input("Enter Stock Symbol", "GOOG")
end = date.today()
start = date(end.year - 20, end.month, end.day)

data = yf.download(stock, start, end)

st.markdown('<h2 class="header">Stock Data</h2>', unsafe_allow_html=True)
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0,1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.markdown('<h3 class="subheader">Price vs Moving Average 100</h3>', unsafe_allow_html=True)
ma_100_days = data['Close'].ewm(span=100, adjust=False).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'm', label='Moving average for 100 days')
plt.plot(data.Close, 'g', label='Closing Price')
plt.legend()
plt.show()
st.pyplot(fig2)

# Prepare the test data
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x, y = np.array(x), np.array(y)

# Predictions
predict_lstm = model_1.predict(x)
predict_gru = model_2.predict(x)
predict_tcn = model_3.predict(x)

scale = 1/scaler.scale_
predict_lstm = predict_lstm * scale
predict_gru = predict_gru * scale
predict_tcn = predict_tcn * scale
y = y * scale

# Plot LSTM predictions
st.markdown('<h2 class="subheader">Prediction using LSTM model</h3>', unsafe_allow_html=True)
fig4 = plt.figure(figsize=(8,6))
plt.plot(y, 'g', label='Original Price')
plt.plot(predict_lstm, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)

# Plot GRU predictions
st.markdown('<h2 class="subheader">Prediction using GRU model</h3>', unsafe_allow_html=True)
fig5 = plt.figure(figsize=(8,6))
plt.plot(y, 'g', label='Original Price')
plt.plot(predict_gru, 'b', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig5)

# Plot TCN predictions
st.markdown('<h2 class="subheader">Prediction using TCN model</h3>', unsafe_allow_html=True)
fig6 = plt.figure(figsize=(8,6))
plt.plot(y, 'g', label='Original Price')
plt.plot(predict_tcn, 'c', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig6)

# Plot all predictions together
st.markdown('<h3 class="subheader">Prection using LSTM, GRU and TCN together </h3>', unsafe_allow_html=True)
fig7 = plt.figure(figsize=(8, 6))
plt.plot(y, 'g', label='Original Price')
plt.plot(predict_lstm, 'r', label='LSTM Predicted Price')
plt.plot(predict_gru, 'b', label='GRU Predicted Price')
plt.plot(predict_tcn, 'orange', label='TCN Predicted Price')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
st.pyplot(fig7)

# Calculate MSE
mse_lstm = mean_squared_error(y, predict_lstm)
mse_gru = mean_squared_error(y, predict_gru)
mse_tcn = mean_squared_error(y, predict_tcn)

# Display MSE
st.markdown('<h3 class="subheader">Model Performance</h3>', unsafe_allow_html=True)
st.markdown(f'<p class="text">LSTM Model MSE: {mse_lstm}</p>', unsafe_allow_html=True)
st.markdown(f'<p class="text">GRU Model MSE: {mse_gru}</p>', unsafe_allow_html=True)
st.markdown(f'<p class="text">TCN Model MSE: {mse_tcn}</p>', unsafe_allow_html=True)

best_model = min(mse_lstm, mse_gru, mse_tcn)
if best_model == mse_lstm:
    best_model_name = "LSTM"
elif best_model == mse_gru:
    best_model_name = "GRU"
else:
    best_model_name = "TCN"

st.markdown(f'<h2 class="header">The most accurate model is: {best_model_name}</h2>', unsafe_allow_html=True)
