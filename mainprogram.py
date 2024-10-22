import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# Set the title for the Streamlit app
st.title('MarketMentorAI - Investment Recommendations')

# Sidebar for user input
st.sidebar.header("Investment Inputs")

# User input for the amount of money to invest
money = st.sidebar.number_input('Investment Amount ($):', min_value=0.0, value=1000.0)

# Input for the investment duration in weeks
time_weeks = st.sidebar.number_input('Investment Duration (weeks):', min_value=1, value=4)

# Risk tolerance input, ranging from 0 (no risk) to 100 (high risk)
risk_tolerance = st.sidebar.number_input('Risk Tolerance (0-100):', min_value=0.0, max_value=100.0, value=50.0)

# Expected return input for user to set their desired returns percentage
expected_returns = st.sidebar.number_input('Expected Returns (%)', min_value=1.0, max_value=100.0, value=10.0)

# Function to clean and flatten multi-level columns
def flatten_columns(df):
    df.columns = [' '.join(col).strip() for col in df.columns.values]
    return df

# Fetch historical stock data from Yahoo Finance
def prepare_data(tickers, start_date, end_date):
    historical_data = []
    for symbol in tickers:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        if not stock_data.empty:
            stock_data = flatten_columns(stock_data)
            stock_data['Symbol'] = symbol
            historical_data.append(stock_data)
    
    if historical_data:
        return pd.concat(historical_data).reset_index()
    else:
        return pd.DataFrame()

# Function for Monte Carlo simulation to project stock prices
def monte_carlo_simulation(start_price, expected_return, risk, num_simulations=1000, num_days=30):
    results = []
    for _ in range(num_simulations):
        price_series = [start_price]
        for _ in range(num_days):
            price = price_series[-1] * np.exp(np.random.normal(expected_return / num_days, risk / np.sqrt(num_days)))
            price_series.append(price)
        results.append(price_series)
    return np.array(results)

# Calculate statistics from Monte Carlo simulation results
def monte_carlo_stats(simulations):
    final_prices = simulations[:, -1]
    avg_final_price = np.mean(final_prices)
    percentage_change = ((avg_final_price - simulations[0, 0]) / simulations[0, 0]) * 100
    profit_or_loss = avg_final_price - simulations[0, 0]
    return avg_final_price, percentage_change, profit_or_loss

# Feature engineering for machine learning model
def prepare_features(data):
    data['Return'] = data['Open'].pct_change()
    data['Lag_1'] = data['Return'].shift(1)
    data['Lag_2'] = data['Return'].shift(2)
    data['Volume_Change'] = data['Volume'].pct_change()
    return data.dropna()

# Train and predict stock prices using Random Forest model
def predict_stock_price(tickers, historical_data):
    predictions = {}
    for symbol in tickers:
        stock_data = historical_data[historical_data['Symbol'] == symbol]
        if not stock_data.empty and 'Open' in stock_data.columns:
            features = prepare_features(stock_data)
            X = features[['Lag_1', 'Lag_2', 'Volume_Change']]
            y = features['Return']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = RandomForestRegressor()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            mse = mean_squared_error(y_test, y_pred)
            predictions[symbol] = {'predicted_returns': y_pred.mean(), 'mse': mse}
    return predictions

# Fetch live data for S&P 500 companies from Wikipedia
@st.cache  # Cache the function to avoid reloading the page every time
def get_live_data_for_companies():
    tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return tickers['Symbol'].tolist()

# Main application logic when the user clicks the 'Predict Stocks' button
if st.button('Predict Stocks'):
    stock_tickers = get_live_data_for_companies()
    end_date = datetime.today()
    start_date = end_date - timedelta(weeks=time_weeks)

    historical_data = prepare_data(stock_tickers, start_date, end_date)

    if historical_data.empty:
        st.error("No historical data available for the selected period.")
    else:
        avg_returns = {}
        for symbol in stock_tickers:
            stock_data = historical_data[historical_data['Symbol'] == symbol]
            if not stock_data.empty:
                daily_returns = stock_data['Open'].pct_change().dropna()
                avg_return = daily_returns.mean()
                risk = daily_returns.std()
                avg_returns[symbol] = (avg_return, risk)

        predictions = predict_stock_price(stock_tickers, historical_data)
        recommended_stocks = []
        for symbol, (avg_return, risk) in avg_returns.items():
            predicted_return = predictions.get(symbol, {}).get('predicted_returns', 0)
            if predicted_return * 100 >= expected_returns and risk * 100 <= risk_tolerance:
                recommended_stocks.append(symbol)

        if recommended_stocks:
            st.subheader("Recommended Stocks:")
            for stock in recommended_stocks:
                st.write(f"### {stock}")

                # Monte Carlo Simulation
                start_price = historical_data[historical_data['Symbol'] == stock]['Open'].iloc[-1]
                sim_results = monte_carlo_simulation(start_price, avg_returns[stock][0], avg_returns[stock][1], num_simulations=1000, num_days=time_weeks)

                avg_final_price, percentage_change, profit_or_loss = monte_carlo_stats(sim_results)
                st.metric(f"Expected Final Price ({time_weeks} weeks)", f"${avg_final_price:.2f}")
                st.metric("Expected Profit/Loss", f"${profit_or_loss:.2f}")
                st.metric("Expected Percentage Change", f"{percentage_change:.2f}%")

                # Plot simulation results
                fig = go.Figure()
                for sim in sim_results:
                    fig.add_trace(go.Scatter(y=sim, mode='lines', line=dict(color='blue', width=1), opacity=0.2))
                fig.update_layout(title=f'Monte Carlo Simulation for {stock}', xaxis_title='Days', yaxis_title='Price', height=400)
                st.plotly_chart(fig)

                # Moving averages for additional analysis
                stock_data = historical_data[historical_data['Symbol'] == stock]
                stock_data['SMA_20'] = stock_data['Open'].rolling(window=20).mean()
                stock_data['SMA_50'] = stock_data['Open'].rolling(window=50).mean()

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Open'], mode='lines', name='Open Price', line=dict(color='orange')))
                fig2.add_trace(go.Scatter(xfig2.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['SMA_20'], mode='lines', name='20 Day SMA', line=dict(color='blue')))
                fig2.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['SMA_50'], mode='lines', name='50 Day SMA', line=dict(color='green')))
                fig2.update_layout(title=f'{stock} Historical Prices with Moving Averages', xaxis_title='Date', yaxis_title='Price', height=400)
                st.plotly_chart(fig2)

                # Value at Risk (VaR) and Conditional Value at Risk (CVaR) calculation
                if not daily_returns.empty:
                    var = np.percentile(daily_returns, 100 - risk_tolerance)
                    cvar = daily_returns[daily_returns <= var].mean()
                    st.write(f"- **Value at Risk (VaR):** {var * 100:.2f}%")
                    st.write(f"- **Conditional Value at Risk (CVaR):** {cvar * 100:.2f}%")
        else:
            st.write("No stocks met the criteria based on your inputs.")

