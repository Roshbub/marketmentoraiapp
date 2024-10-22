import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm

# Title of the app
st.title('MarketMentorAI')

# Sidebar for user inputs
st.sidebar.header("Investment Inputs")
money = st.sidebar.number_input('Enter the amount of money:', min_value=0.0, value=1000.0)
time = st.sidebar.number_input('Enter the time in weeks:', min_value=1, value=4)
risk_percentage = st.sidebar.number_input('Enter risk percentage (0-100):', min_value=0.0, max_value=100.0, value=50.0)
returns = st.sidebar.number_input('Enter expected returns (1-100):', min_value=1.0, max_value=100.0, value=10.0)

# Dropdown for historical data period
valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
historical_period = st.sidebar.selectbox('Select historical data period:', valid_periods)

# Global variables to hold fetched data
historical_data = pd.DataFrame()
predicted_returns = pd.Series()

def flatten_columns(df):
    """Flatten MultiIndex columns into single level."""
    df.columns = [' '.join(col).strip() for col in df.columns.values]
    return df

def prepare_data(tickers, period):
    """Fetch historical stock data."""
    historical_data = []
    with st.spinner('Downloading historical data...'):
        for symbol in tqdm(tickers['Symbol'], desc="Downloading data"):
            try:
                stock_data = yf.download(symbol, period=period)
                stock_data = flatten_columns(stock_data)  # Flatten the MultiIndex
                if not stock_data.empty:
                    stock_data['Symbol'] = symbol
                    historical_data.append(stock_data)
            except Exception as e:
                st.warning(f"Could not download data for {symbol}: {e}")

    combined_data = pd.concat(historical_data) if historical_data else pd.DataFrame()
    return combined_data.reset_index()  # Resetting index can also help with fragmentation

# Main application logic
if st.button('Predict Stocks'):
    try:
        tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    except Exception as e:
        st.error(f"Error fetching S&P 500 companies: {e}")
        tickers = pd.DataFrame()

    if not tickers.empty:
        historical_data = prepare_data(tickers, historical_period)
        if not historical_data.empty:
            historical_data['year'] = historical_data['Date'].dt.year
            historical_data['month'] = historical_data['Date'].dt.month
            historical_data['day'] = historical_data['Date'].dt.day

            # Feature Engineering
            # Collect columns to add to a new DataFrame instead of inserting one at a time
            feature_columns = [
                col for col in historical_data.columns 
                if col not in ['Symbol', 'Date', 'year', 'month', 'day'] and historical_data[col].dtype in [np.float64, np.int64]
            ]
            
            # Create a copy of the DataFrame with necessary columns
            le = LabelEncoder()
            historical_data['symbol_encoded'] = le.fit_transform(historical_data['Symbol'])
            features = historical_data[['symbol_encoded', 'year', 'month', 'day'] + feature_columns].copy()

            # Assign target after checking for existence
            target = historical_data['Close'] if 'Close' in historical_data.columns else pd.Series(dtype=float)

            if not target.empty:
                X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

                # Model Training
                rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_regressor.fit(X_train, y_train)

                # Evaluation
                y_pred = rf_regressor.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                st.write(f"Mean Squared Error on Test Set: {mse}")

                # Simulated live data
                live_data_sp500 = {symbol: {'Open': 100, 'High': 105, 'Low': 95, 'Close': 100, 'Volume': 1000} for symbol in tickers['Symbol']}
                predicted_returns = predict_stock_investment(rf_regressor, live_data_sp500, le)
                display_predictions(predicted_returns)

                # Recommend stocks based on user input
                recommended_stocks = recommend_stocks(live_data_sp500, predicted_returns, risk_percentage, money)
                display_recommendations(recommended_stocks, money)
