import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Load models
model_1 = load_model("LSTM_Stock Prediction Model.h5")
model_2 = load_model("GRU_Stock_Prediction_Model.h5")
model_3 = load_model("TCN_Stock_Prediction_Model.h5")

# Streamlit Application
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

# Function for Holt's Two-Parameter Smoothing
def holt_two_parameter_smoothing(series, alpha, beta):
    smoothed_series = [series[0]]  # Initial smoothed value
    trend = [series[1] - series[0]]  # Initial trend estimate

    for t in range(1, len(series)):
        smoothed_value = alpha * series[t] + (1 - alpha) * (smoothed_series[t-1] + trend[t-1])
        new_trend = beta * (smoothed_value - smoothed_series[t-1]) + (1 - beta) * trend[t-1]
        
        smoothed_series.append(smoothed_value)
        trend.append(new_trend)

    return np.array(smoothed_series)

# Function to calculate MSE, MAE, RMSE, and MAPE with non-overlapping windows of 'span' days
def holt_smoothing_metrics(predict, y_true, alpha, beta, span):
    smoothed_predictions = holt_two_parameter_smoothing(predict.flatten(), alpha, beta)
    
    mse_sum = mae_sum = mape_sum = 0.0
    num_samples = 0
    
    # Calculate metrics for non-overlapping windows of 'span' days
    for i in range(0, len(y_true) - span, span):
        y_true_subset = y_true[i:i+span]
        smoothed_subset = smoothed_predictions[i:i+span]
        
        mse_sum += mean_squared_error(y_true_subset, smoothed_subset)
        mae_sum += mean_absolute_error(y_true_subset, smoothed_subset)
        mape_sum += mean_absolute_percentage_error(y_true_subset, smoothed_subset)
        num_samples += 1
    
    # Calculate metrics for the last window that might not be of length 'span'
    if len(y_true) % span != 0:
        y_true_subset = y_true[-(len(y_true) % span):]
        smoothed_subset = smoothed_predictions[-(len(y_true) % span):]
        
        mse_sum += mean_squared_error(y_true_subset, smoothed_subset)
        mae_sum += mean_absolute_error(y_true_subset, smoothed_subset)
        mape_sum += mean_absolute_percentage_error(y_true_subset, smoothed_subset)
        num_samples += 1
    
    if num_samples > 0:
        average_mse = mse_sum / num_samples
        average_mae = mae_sum / num_samples
        average_mape = mape_sum / num_samples
    else:
        average_mse = average_mae = average_mape = 0.0
    
    return average_mse, np.sqrt(average_mse), average_mae, average_mape

st.markdown('<h1 class="title">Stock Market Predictor</h1>', unsafe_allow_html=True)
stock = st.text_input("Enter Stock Symbol", "GOOG").upper()
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

# Plot Moving Average and Holt Smoothing
st.markdown('<h3 class="subheader">Price vs Single Parameter Moving Average for 100 days</h3>', unsafe_allow_html=True)
ma_100_days = data['Close'].ewm(span=100, adjust=False).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'm', label='Moving average for 100 days')
plt.plot(data.Close, 'g', label='Closing Price')
plt.legend()
plt.show()
st.pyplot(fig2)

# Plot Holt's smoothing vs closing price
st.markdown('<h3 class="subheader">Price vs Holt Smoothing Double Parameter Moving Average</h3>', unsafe_allow_html=True)
alpha_value = 0.05  # Adjust alpha as needed
beta_value = 0.05  # Adjust beta as needed
holt_smoothing = holt_two_parameter_smoothing(data['Close'].values, alpha_value, beta_value)
fig3 = plt.figure(figsize=(8, 6))
plt.plot(data.index, data.Close, 'g', label='Closing Price')
plt.plot(data.index, holt_smoothing, 'm', label='Holt Smoothing')
plt.legend()
plt.show()
st.pyplot(fig3)

# Prepare the test data for predictions
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x, y = np.array(x), np.array(y)

# Predictions using loaded models
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
plt.plot(predict_tcn, 'orange', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig6)

# Plot all predictions together
st.markdown('<h3 class="subheader">Prediction using LSTM, GRU and TCN together </h3>', unsafe_allow_html=True)
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

# Function to compute Double Parameter Exponential Moving Average (Holt Smoothing) metrics
def double_param_ma_metrics(predict, y_true, alpha, beta, span):
    return holt_smoothing_metrics(predict, y_true, alpha, beta, span)

# Calculate metrics using Holt's method with gaps of 50, 100, and 150 days
alpha_value = 0.05  # Adjust alpha as needed
beta_value = 0.05  # Adjust beta as needed

mse_lstm_50_da, rmse_lstm_50_da, mae_lstm_50_da, mape_lstm_50_da = double_param_ma_metrics(predict_lstm, y, alpha_value, beta_value, 50)
mse_lstm_100_da, rmse_lstm_100_da, mae_lstm_100_da, mape_lstm_100_da = double_param_ma_metrics(predict_lstm, y, alpha_value, beta_value, 100)
mse_lstm_150_da, rmse_lstm_150_da, mae_lstm_150_da, mape_lstm_150_da = double_param_ma_metrics(predict_lstm, y, alpha_value, beta_value, 150)

mse_gru_50_da, rmse_gru_50_da, mae_gru_50_da, mape_gru_50_da = double_param_ma_metrics(predict_gru, y, alpha_value, beta_value, 50)
mse_gru_100_da, rmse_gru_100_da, mae_gru_100_da, mape_gru_100_da = double_param_ma_metrics(predict_gru, y, alpha_value, beta_value, 100)
mse_gru_150_da, rmse_gru_150_da, mae_gru_150_da, mape_gru_150_da = double_param_ma_metrics(predict_gru, y, alpha_value, beta_value, 150)

mse_tcn_50_da, rmse_tcn_50_da, mae_tcn_50_da, mape_tcn_50_da = double_param_ma_metrics(predict_tcn, y, alpha_value, beta_value, 50)
mse_tcn_100_da, rmse_tcn_100_da, mae_tcn_100_da, mape_tcn_100_da = double_param_ma_metrics(predict_tcn, y, alpha_value, beta_value, 100)
mse_tcn_150_da, rmse_tcn_150_da, mae_tcn_150_da, mape_tcn_150_da = double_param_ma_metrics(predict_tcn, y, alpha_value, beta_value, 150)

# Calculate metrics for individual models without double parameter smoothing
mse_lstm = mean_squared_error(y, predict_lstm)
rmse_lstm = np.sqrt(mse_lstm)
mae_lstm = mean_absolute_error(y, predict_lstm)
mape_lstm = mean_absolute_percentage_error(y, predict_lstm)

mse_gru = mean_squared_error(y, predict_gru)
rmse_gru = np.sqrt(mse_gru)
mae_gru = mean_absolute_error(y, predict_gru)
mape_gru = mean_absolute_percentage_error(y, predict_gru)

mse_tcn = mean_squared_error(y, predict_tcn)
rmse_tcn = np.sqrt(mse_tcn)
mae_tcn = mean_absolute_error(y, predict_tcn)
mape_tcn = mean_absolute_percentage_error(y, predict_tcn)

def double_param_ma(span, model):
    alpha_value = 2/(1+span) 
    beta_value = 1 - alpha_value
    first_ewm = model.ewm(alpha=alpha_value, adjust=False).mean()
    double_parameter_MA = first_ewm.ewm(alpha=beta_value, adjust=False).mean()
    return double_parameter_MA

ma_lstm_50 = double_param_ma(50, pd.Series(predict_lstm.flatten()))
ma_lstm_100 = double_param_ma(100, pd.Series(predict_lstm.flatten()))
ma_lstm_150 = double_param_ma(150, pd.Series(predict_lstm.flatten()))

ma_gru_50 = double_param_ma(50, pd.Series(predict_gru.flatten()))
ma_gru_100 = double_param_ma(100, pd.Series(predict_gru.flatten()))
ma_gru_150 = double_param_ma(150, pd.Series(predict_gru.flatten()))

ma_tcn_50 = double_param_ma(50, pd.Series(predict_tcn.flatten()))
ma_tcn_100 = double_param_ma(100, pd.Series(predict_tcn.flatten()))
ma_tcn_150 = double_param_ma(150, pd.Series(predict_tcn.flatten()))

mse_lstm_50 = mean_squared_error(y, ma_lstm_50)
mae_lstm_50 = mean_absolute_error(y, ma_lstm_50)
rmse_lstm_50 = np.sqrt(mse_lstm_50)
mape_lstm_50 = mean_absolute_percentage_error(y, ma_lstm_50)

mse_lstm_100 = mean_squared_error(y, ma_lstm_100)
mae_lstm_100 = mean_absolute_error(y, ma_lstm_100)
rmse_lstm_100 = np.sqrt(mse_lstm_100)
mape_lstm_100 = mean_absolute_percentage_error(y, ma_lstm_100)

mse_lstm_150 = mean_squared_error(y, ma_lstm_150)
mae_lstm_150 = mean_absolute_error(y, ma_lstm_150)
rmse_lstm_150 = np.sqrt(mse_lstm_150)
mape_lstm_150 = mean_absolute_percentage_error(y, ma_lstm_150)

mse_gru_50 = mean_squared_error(y, ma_gru_50)
mae_gru_50 = mean_absolute_error(y, ma_gru_50)
rmse_gru_50 = np.sqrt(mse_gru_50)
mape_gru_50 = mean_absolute_percentage_error(y, ma_gru_50)

mse_gru_100 = mean_squared_error(y, ma_gru_100)
mae_gru_100 = mean_absolute_error(y, ma_gru_100)
rmse_gru_100 = np.sqrt(mse_gru_100)
mape_gru_100 = mean_absolute_percentage_error(y, ma_gru_100)

mse_gru_150 = mean_squared_error(y, ma_gru_150)
mae_gru_150 = mean_absolute_error(y, ma_gru_150)
rmse_gru_150 = np.sqrt(mse_gru_150)
mape_gru_150 = mean_absolute_percentage_error(y, ma_gru_150)

mse_tcn_50 = mean_squared_error(y, ma_tcn_50)
mae_tcn_50 = mean_absolute_error(y, ma_tcn_50)
rmse_tcn_50 = np.sqrt(mse_tcn_50)
mape_tcn_50 = mean_absolute_percentage_error(y, ma_tcn_50)

mse_tcn_100 = mean_squared_error(y, ma_tcn_100)
mae_tcn_100 = mean_absolute_error(y, ma_tcn_100)
rmse_tcn_100 = np.sqrt(mse_tcn_100)
mape_tcn_100 = mean_absolute_percentage_error(y, ma_tcn_100)

mse_tcn_150 = mean_squared_error(y, ma_tcn_150)
mae_tcn_150 = mean_absolute_error(y, ma_tcn_150)
rmse_tcn_150 = np.sqrt(mse_tcn_150)
mape_tcn_150 = mean_absolute_percentage_error(y, ma_tcn_150)

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mae, rmse, mape

st.markdown('<h3 class="subheader">MSE, MAE, RMSE, and MAPE Comparison for Models taking span = 1 day</h3>', unsafe_allow_html=True)

output_table_normal = {
    'Model': ['LSTM', 'GRU', 'TCN'],
    'MSE': [mse_lstm, mse_gru, mse_tcn],
    'MAE': [mae_lstm, mae_gru, mae_tcn],
    'RMSE': [rmse_lstm, rmse_gru, rmse_tcn],
    'MAPE': [mape_lstm, mape_gru, mape_tcn]
}
df_normal = pd.DataFrame(output_table_normal)
st.table(df_normal)

# Display metrics comparison using Single Parameter Exponential Moving Average
st.markdown('<h3 class="subheader">MSE Comparison using Single Parameter Exponential Moving Average</h3>', unsafe_allow_html=True)

output_table_single_param_1 = {
    'Model': ['LSTM', 'GRU', 'TCN'],
    'MSE (Span = 50 days)': [mse_lstm_50, mse_gru_50, mse_tcn_50],
    'MSE (Span = 100 days)': [mse_lstm_100, mse_gru_100, mse_tcn_100],
    'MSE (Span = 150 days)': [mse_lstm_150, mse_gru_150, mse_tcn_150],
}
df_single_param_1 = pd.DataFrame(output_table_single_param_1)
st.table(df_single_param_1)

st.markdown('<h3 class="subheader">MAE Comparison using Single Parameter Exponential Moving Average</h3>', unsafe_allow_html=True)
output_table_single_param_2 = {
    'Model': ['LSTM', 'GRU', 'TCN'],
    'MAE (Span = 50 days)': [mae_lstm_50, mae_gru_50, mae_tcn_50],
    'MAE (Span = 100 days)': [mae_lstm_100, mae_gru_100, mae_tcn_100],
    'MAE (Span = 150 days)': [mae_lstm_150, mae_gru_150, mae_tcn_150],
}
df_single_param_2 = pd.DataFrame(output_table_single_param_2)
st.table(df_single_param_2)

st.markdown('<h3 class="subheader">RMSE Comparison using Single Parameter Exponential Moving Average</h3>', unsafe_allow_html=True)
output_table_single_param_3 = {
    'Model': ['LSTM', 'GRU', 'TCN'],
    'RMSE (Span = 50 days)': [rmse_lstm_50, rmse_gru_50, rmse_tcn_50],
    'RMSE (Span = 100 days)': [rmse_lstm_100, rmse_gru_100, rmse_tcn_100],
    'RMSE (Span = 150 days)': [rmse_lstm_150, rmse_gru_150, rmse_tcn_150],
}
df_single_param_3 = pd.DataFrame(output_table_single_param_3)
st.table(df_single_param_3)

st.markdown('<h3 class="subheader">MAPE Comparison using Single Parameter Exponential Moving Average</h3>', unsafe_allow_html=True)
output_table_single_param_4 = {
    'Model': ['LSTM', 'GRU', 'TCN'],
    'MAPE (Span = 50 days)': [mape_lstm_50, mape_gru_50, mape_tcn_50],
    'MAPE (Span = 100 days)': [mape_lstm_100, mape_gru_100, mape_tcn_100],
    'MAPE (Span = 150 days)': [mape_lstm_150, mape_gru_150, mape_tcn_150],
}
df_single_param_4 = pd.DataFrame(output_table_single_param_4)
st.table(df_single_param_4)

# Display metrics comparison using Holt's Double Parameter Smoothing Method
st.markdown('<h3 class="subheader">MSE Comparison using Holt\'s Double Parameter Smoothing Method</h3>', unsafe_allow_html=True)

output_table_double_param_1 = {
    'Model': ['LSTM', 'GRU', 'TCN'],
    'MSE (Span = 50 days)': [mse_lstm_50_da, mse_gru_50_da, mse_tcn_50_da],
    'MSE (Span = 100 days)': [mse_lstm_100_da, mse_gru_100_da, mse_tcn_100_da],
    'MSE (Span = 150 days)': [mse_lstm_150_da, mse_gru_150_da, mse_tcn_150_da],
}
df_double_param_1 = pd.DataFrame(output_table_double_param_1)
st.table(df_double_param_1)

st.markdown('<h3 class="subheader">MAE Comparison using Holt\'s Double Parameter Smoothing Method</h3>', unsafe_allow_html=True)

output_table_double_param_2 = {
    'Model': ['LSTM', 'GRU', 'TCN'],
    'MAE (Span = 50 days)': [mae_lstm_50_da, mae_gru_50_da, mae_tcn_50_da],
    'MAE (Span = 100 days)': [mae_lstm_100_da, mae_gru_100_da, mae_tcn_100_da],
    'MAE (Span = 150 days)': [mae_lstm_150_da, mae_gru_150_da, mae_tcn_150_da],
}
df_double_param_2 = pd.DataFrame(output_table_double_param_2)
st.table(df_double_param_2)

st.markdown('<h3 class="subheader">RMSE Comparison using Holt\'s Double Parameter Smoothing Method</h3>', unsafe_allow_html=True)

output_table_double_param_3 = {
    'Model': ['LSTM', 'GRU', 'TCN'],
    'RMSE (Span = 50 days)': [rmse_lstm_50_da, rmse_gru_50_da, rmse_tcn_50_da],
    'RMSE (Span = 100 days)': [rmse_lstm_100_da, rmse_gru_100_da, rmse_tcn_100_da],
    'RMSE (Span = 150 days)': [rmse_lstm_150_da, rmse_gru_150_da, rmse_tcn_150_da],
}
df_double_param_3 = pd.DataFrame(output_table_double_param_3)
st.table(df_double_param_3)

st.markdown('<h3 class="subheader">MAPE Comparison using Holt\'s Double Parameter Smoothing Method</h3>', unsafe_allow_html=True)

output_table_double_param_4 = {
    'Model': ['LSTM', 'GRU', 'TCN'],
    'MAPE (Span = 50 days)': [mape_lstm_50_da, mape_gru_50_da, mape_tcn_50_da],
    'MAPE (Span = 100 days)': [mape_lstm_100_da, mape_gru_100_da, mape_tcn_100_da],
    'MAPE (Span = 150 days)': [mape_lstm_150_da, mape_gru_150_da, mape_tcn_150_da],
}
df_double_param_4 = pd.DataFrame(output_table_double_param_4)
st.table(df_double_param_4)

# Determine the best model based on the lowest MSE
mse_values = [mse_lstm, mse_gru, mse_tcn]
best_model_idx = np.argmin(mse_values)
best_model_name = output_table_normal['Model'][best_model_idx]
best_model_mse = mse_values[best_model_idx]

# Display the best model and its MSE
st.markdown(f'<h2 class="header">The most accurate model is: {best_model_name}</h2>', unsafe_allow_html=True)
