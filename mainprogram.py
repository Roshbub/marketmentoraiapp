import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import requests
from datetime import datetime, timedelta

# Title of the app
st.title('MarketMentorAI')

# Sidebar for user inputs
st.sidebar.header("Investment Inputs")
money = st.sidebar.number_input('Enter the amount of money:', min_value=0.0, value=1000.0)
time_weeks = st.sidebar.number_input('Enter the time in weeks:', min_value=1, value=4)
risk_percentage = st.sidebar.number_input('Enter risk percentage (0-100):', min_value=0.0, max_value=100.0, value=50.0)
returns = st.sidebar.number_input('Enter expected returns (1-100):', min_value=1.0, max_value=100.0, value=10.0)

# Function to flatten MultiIndex columns into a single level
def flatten_columns(df):
    df.columns = [' '.join(col).strip() for col in df.columns.values]
    return df

# Function to fetch historical stock data
def prepare_data(tickers, start_date, end_date):
    historical_data = []
    for symbol in tickers:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        if not stock_data.empty:
            stock_data = flatten_columns(stock_data)  # Flatten the MultiIndex
            stock_data['Symbol'] = symbol
            historical_data.append(stock_data)

    combined_data = pd.concat(historical_data) if historical_data else pd.DataFrame()
    return combined_data.reset_index()

# Function for Monte Carlo Simulation
def monte_carlo_simulation(start_price, expected_return, risk, num_simulations=1000, num_days=30):
    results = []
    for _ in range(num_simulations):
        price_series = [start_price]
        for _ in range(num_days):
            price = price_series[-1] * np.exp(np.random.normal(expected_return / num_days, risk / np.sqrt(num_days)))
            price_series.append(price)
        results.append(price_series)
    return np.array(results)

# Function for fetching news articles
def fetch_news(ticker):
    news_api_url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey=YOUR_NEWS_API_KEY"
    response = requests.get(news_api_url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        return articles
    else:
        return []  # Return empty list on error

# Function for Feature Engineering for ML Model
def prepare_features(data):
    data['Return'] = data['Open'].pct_change()  # Use 'Open' price for returns
    data['Lag_1'] = data['Return'].shift(1)
    data['Lag_2'] = data['Return'].shift(2)
    data['Volume_Change'] = data['Volume'].pct_change()
    data = data.dropna()  # Drop rows with NaN values
    return data

# Function for Model Prediction
def predict_stock_price(tickers, historical_data):
    predictions = {}
    for symbol in tickers:
        stock_data = historical_data[historical_data['Symbol'] == symbol]
        if not stock_data.empty and 'Open' in stock_data.columns:
            features = prepare_features(stock_data)
            X = features[['Lag_1', 'Lag_2', 'Volume_Change']]
            y = features['Return']
            
            # Splitting data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scaling features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Fitting Random Forest Regressor
            model = RandomForestRegressor()
            model.fit(X_train_scaled, y_train)

            # Making predictions
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)

            predictions[symbol] = {'predicted_returns': y_pred.mean(), 'mse': mse}
    return predictions

# Function to get live data for S&P 500 companies
def get_live_data_for_companies():
    tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return tickers['Symbol'].tolist()

# Main application logic
if st.button('Predict Stocks'):
    stock_tickers = get_live_data_for_companies()

    # Calculate date range based on user input
    end_date = datetime.today()
    start_date = end_date - timedelta(weeks=time_weeks)

    historical_data = prepare_data(stock_tickers, start_date, end_date)

    # Check if historical_data is empty
    if historical_data.empty:
        st.write("No historical data available for the selected period.")
    else:
        # Check for necessary columns
        if 'Open' not in historical_data.columns or 'Volume' not in historical_data.columns:
            st.write("The required data columns are not available in the historical data.")
        else:
            # Calculate average return and risk for each stock
            avg_returns = {}
            for symbol in stock_tickers:
                stock_data = historical_data[historical_data['Symbol'] == symbol]
                if not stock_data.empty and 'Open' in stock_data.columns:
                    daily_returns = stock_data['Open'].pct_change().dropna()  # Use 'Open' for daily returns
                    avg_return = daily_returns.mean()
                    risk = daily_returns.std()
                    avg_returns[symbol] = (avg_return, risk)

            # Predict future stock returns using the ML model
            predictions = predict_stock_price(stock_tickers, historical_data)

            # Filter stocks based on user inputs
            recommended_stocks = []
            for symbol, (avg_return, risk) in avg_returns.items():
                predicted_return = predictions.get(symbol, {}).get('predicted_returns', 0)
                if predicted_return * 100 >= returns and risk * 100 <= risk_percentage:
                    recommended_stocks.append(symbol)

            if recommended_stocks:
                st.subheader("Recommended Stocks:")
                for stock in recommended_stocks:
                    st.write(f"### {stock}")

                    # Monte Carlo Simulation
                    start_price = historical_data[historical_data['Symbol'] == stock]['Open'].iloc[-1]  # Use 'Open' price
                    sim_results = monte_carlo_simulation(start_price, avg_returns[stock][0], avg_returns[stock][1], num_simulations=1000, num_days=time_weeks)

                    # Plotting simulation results
                    fig = go.Figure()
                    for sim in sim_results:
                        fig.add_trace(go.Scatter(y=sim, mode='lines', name=stock, showlegend=False, line=dict(color='blue', width=1), opacity=0.2))
                    fig.update_layout(title=f'Monte Carlo Simulation for {stock}', xaxis_title='Days', yaxis_title='Price', height=400)
                    st.plotly_chart(fig)

                    # Fetch news articles
                    news_articles = fetch_news(stock)
                    st.write(f"### Why Invest in {stock}:")
                    if news_articles:
                        for article in news_articles[:3]:  # Display top 3 news articles
                            st.write(f"- **{article['title']}**")
                            st.write(f"  *{article['description']}*")
                            st.write(f"[Read more]({article['url']})")
                    else:
                        st.write("No recent news available.")

                    # Advanced stock analysis (e.g., moving averages)
                    stock_data = historical_data[historical_data['Symbol'] == stock]
                    stock_data['SMA_20'] = stock_data['Open'].rolling(window=20).mean()  # Using 'Open' for SMA
                    stock_data['SMA_50'] = stock_data['Open'].rolling(window=50).mean()  # Using 'Open' for SMA

                    # Plotting historical prices with moving averages
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Open'], mode='lines', name='Open Price', line=dict(color='orange', width=2)))
                    fig2.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['SMA_20'], mode='lines', name='20 Day SMA', line=dict(color='blue', width=2)))
                    fig2.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['SMA_50'], mode='lines', name='50 Day SMA', line=dict(color='green', width=2)))
                    fig2.update_layout(title=f'{stock} Historical Prices with Moving Averages', xaxis_title='Date', yaxis_title='Price', height=400)
                    st.plotly_chart(fig2)

                    # Value at Risk calculation
                    if not daily_returns.empty:
                        var = np.percent
                        var = np.percentile(daily_returns, 100 - risk_percentage)
                        cvar = daily_returns[daily_returns <= var].mean()
                        st.write(f"- **Value at Risk (VaR)**: {var * 100:.2f}%")
                        st.write(f"- **Conditional Value at Risk (CVaR)**: {cvar * 100:.2f}%")
       

