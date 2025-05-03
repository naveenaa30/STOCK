import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objects as go  # Import Plotly for enhanced visualization

# Load trained LSTM model
custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}
model = load_model("lstm_stock_model.h5", custom_objects=custom_objects)

# Fetch stock data
def get_stock_data(ticker, start_date, end_date):
    stock = yf.download(ticker, start=start_date, end=end_date)

    # Compute RSI
    delta = stock['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    stock['RSI'] = 100 - (100 / (1 + rs))

    # Compute MACD
    stock['EMA_12'] = stock['Close'].ewm(span=12, adjust=False).mean()
    stock['EMA_26'] = stock['Close'].ewm(span=26, adjust=False).mean()
    stock['MACD'] = stock['EMA_12'] - stock['EMA_26']

    # Compute Bollinger Bands
    stock['Bollinger_Upper'] = stock['Close'].rolling(window=20).mean() + (2 * stock['Close'].rolling(window=20).std())
    stock['Bollinger_Lower'] = stock['Close'].rolling(window=20).mean() - (2 * stock['Close'].rolling(window=20).std())

    stock = stock[['Close', 'RSI', 'MACD', 'Bollinger_Upper', 'Bollinger_Lower']].dropna()
    return stock

# Preprocess Data
def preprocess_data(df, seq_length=30):
    scaler = MinMaxScaler(feature_range=(0.05, 0.95))
    scaled_data = scaler.fit_transform(df)
    return np.array(scaled_data), scaler

# Predict next 30 days dynamically
def predict_next_30_days(model, last_sequence, scaler):
    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(30):
        predicted_scaled = model.predict(np.expand_dims(current_sequence, axis=(0, -1)))  
        future_predictions.append(predicted_scaled[0][0])

        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1, 0] = predicted_scaled

    placeholder_array = np.zeros((30, scaler.n_features_in_))
    placeholder_array[:, 0] = future_predictions

    return scaler.inverse_transform(placeholder_array)[:, 0]

# Streamlit UI
st.title("Stock Price 30-Day Predictor (Improved)")

ticker = st.text_input("Enter Stock Ticker:", "AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2025-04-01"))

if st.button("Fetch Data & Predict"):
    df = get_stock_data(ticker, start_date, end_date)

    if df.empty:
        st.error("No data found. Try a different ticker or date range.")
        st.stop()

    st.line_chart(df['Close'])

    scaled_data, scaler = preprocess_data(df)
    last_sequence = scaled_data[-30:].reshape(-1, scaled_data.shape[1])

    future_prices = predict_next_30_days(model, last_sequence, scaler)

    # Display predictions
    st.subheader(f"Predicted Next 30 Days for {ticker}:")
    future_df = pd.DataFrame({"Day": range(1, 31), "Predicted Price": future_prices})
    st.write(future_df)

    # **Enhanced Line Chart for Clear Visualization**
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=future_df["Day"],
        y=future_df["Predicted Price"],
        mode="lines+markers",  # Smooth line with markers for clarity
        name="Predicted Prices",
        line=dict(color="blue", width=3)  # Better visibility with bold lines
    ))

    fig.update_layout(
        title=f"{ticker} Stock Price Prediction",
        xaxis_title="Days Ahead",
        yaxis_title="Predicted Stock Price",
        template="plotly_white",  # Light background for better contrast
        xaxis=dict(showgrid=True),  # Gridlines for easier trend interpretation
        yaxis=dict(showgrid=True),
        font=dict(size=12)
    )

    st.plotly_chart(fig)
