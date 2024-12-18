# Import Libraries
import pandas as pd
import numpy as np
import requests
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Function to Fetch Crypto Prices with Fallback Data
def fetch_prices(crypto_ids=["bitcoin", "ethereum", "solana"], days="60"):
    crypto_data = {}
    try:
        for crypto in crypto_ids:
            url = f"https://api.coingecko.com/api/v3/coins/{crypto}/market_chart"
            params = {"vs_currency": "usd", "days": days}
            response = requests.get(url, params=params)
            if response.status_code == 200:
                prices = [item[1] for item in response.json()["prices"]]
                crypto_data[crypto] = prices
            else:
                raise Exception("Primary API failed")
    except:
        # Fallback Data
        crypto_data = {
            "bitcoin": [50000, 51000, 52000, 53000, 54000, 55000],
            "ethereum": [3000, 3050, 3100, 3200, 3300, 3400],
            "solana": [150, 160, 170, 180, 190, 200]
        }
    return crypto_data

# Machine Learning Model for Price Prediction
def train_predict(prices):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(np.array(prices).reshape(-1, 1))

    X_train, y_train = [], []
    for i in range(60, len(scaled_prices)):
        X_train.append(scaled_prices[i-60:i, 0])
        y_train.append(scaled_prices[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    # Predict Next Price
    last_60_days = scaled_prices[-60:].reshape(1, 60, 1)
    predicted_price = model.predict(last_60_days)
    return scaler.inverse_transform(predicted_price)[0][0]

# Streamlit UI for IWE
def main():
    st.title("🌟 Infinite Wealth Ecosystem 🌟")
    st.subheader("Multi-Asset Yield | AI-Powered Predictions | Portfolio Optimization")

    # Section 1: Fetch Crypto Prices
    st.write("### 1. Fetching Real-Time Prices for BTC, ETH, and SOL")
    crypto_data = fetch_prices()
    
    st.write("#### Latest Prices (Fallback Used if API Failed):")
    for crypto, prices in crypto_data.items():
        st.write(f"**{crypto.capitalize()}**: ${prices[-1]:,.2f}")

    # Section 2: AI-Powered Price Prediction
    st.write("### 2. AI-Powered Price Prediction")
    predicted_prices = {}
    for crypto, prices in crypto_data.items():
        st.write(f"Training AI for {crypto.capitalize()}...")
        predicted_prices[crypto] = train_predict(prices)
    
    st.write("#### Predicted Next Day Prices:")
    for crypto, pred in predicted_prices.items():
        st.write(f"**{crypto.capitalize()}**: ${pred:,.2f}")

    # Section 3: Portfolio and Interest Optimization
    st.write("### 3. Portfolio and Interest Optimization")
    total_portfolio = st.number_input("Enter your Total Portfolio Value (USD):", value=10000)
    expected_rate = st.slider("Expected Annual Interest Rate (%)", 1, 20, 10)
    payout_freq = st.radio("Choose Payout Frequency:", ["Monthly", "Annually"])

    annual_interest = total_portfolio * (expected_rate / 100)
    payout = annual_interest / 12 if payout_freq == "Monthly" else annual_interest
    st.write(f"**Your {payout_freq} Payout**: ${payout:,.2f}")

    # Section 4: Visualize Historical Prices
    st.write("### 4. Historical Price Visualization")
    for crypto, prices in crypto_data.items():
        st.write(f"**{crypto.capitalize()} Price History**")
        plt.figure(figsize=(10, 4))
        plt.plot(prices, label=f"{crypto.capitalize()} Prices", color="blue")
        plt.xlabel("Days")
        plt.ylabel("Price (USD)")
        plt.legend()
        st.pyplot(plt)

    st.write("🚀 **Infinite Wealth Ecosystem** brings next-level AI-powered insights, ensuring financial success and optimized yields!")

# Run the App
if __name__ == "__main__":
    main()
