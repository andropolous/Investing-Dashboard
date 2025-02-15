import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Function to fetch stock data from Alpha Vantage API
def get_stock_data(ticker, period='6mo'):
    api_key = "YOUR_ALPHAVANTAGE_API_KEY"
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}&outputsize=compact"
    try:
        response = requests.get(url)
        data = response.json()
        time_series = data.get("Time Series (Daily)", {})
        df = pd.DataFrame(time_series).T.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume"
        })
        df = df.sort_index()
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# Function to plot stock price and moving averages
def plot_stock_data(df, ticker):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df.index, df['Close'], label='Close Price', linewidth=2)
    ax.plot(df.index, df['SMA_50'], label='50-day SMA', linestyle='dashed')
    ax.plot(df.index, df['SMA_200'], label='200-day SMA', linestyle='dotted')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.set_title(f'{ticker} - Stock Price & Moving Averages')
    ax.legend()
    ax.grid()
    st.pyplot(fig)

# AI-Powered Stock Price Prediction
def predict_price(df):
    df['Days'] = (df.index - df.index[0]).days
    X = df[['Days']].values.reshape(-1, 1)  # Ensure X is properly formatted
    y = df['Close'].values.reshape(-1, 1)  # Ensure y is properly formatted
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_days = np.array([(df.index[-1] - df.index[0]).days + i for i in range(1, 31)]).reshape(-1, 1)
    future_prices = model.predict(future_days)
    
    future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, 31)]
    
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df.index, df['Close'], label='Historical Prices')
    ax.plot(future_dates, future_prices, linestyle='dashed', label='Predicted Prices', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.set_title('Stock Price Prediction')
    ax.legend()
    ax.grid()
    st.pyplot(fig)

# Streamlit UI
st.title("📈 AI-Powered Stock Investing Dashboard")

# User Input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, NVDA):", "AAPL").upper()

if ticker:
    df = get_stock_data(ticker)

    if not df.empty:
        st.subheader(f"Stock Data for {ticker}")
        st.dataframe(df.tail())
        
        # Plot stock data with moving averages
        plot_stock_data(df, ticker)
        
        # Predict Future Prices
        st.subheader("📊 AI-Powered Stock Price Prediction")
        predict_price(df)
        
        # Stock alerts
        latest_price = df['Close'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        
        if latest_price < sma_50:
            st.warning(f'🚨 {ticker} is trading below the 50-day SMA. Possible buying opportunity!')
        else:
            st.success(f'✅ {ticker} is above the 50-day SMA. No action needed.')
    else:
        st.error("Invalid stock ticker. Please enter a valid symbol.")
