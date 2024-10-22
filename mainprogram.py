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

def prepare_data(tickers, period):
    """Fetch historical stock data."""
    historical_data = []
    with st.spinner('Downloading historical data...'):
        for symbol in tqdm(tickers['Symbol'], desc="Downloading data"):
            try:
                stock_data = yf.download(symbol, period=period)
                if not stock_data.empty:
                    stock_data['Symbol'] = symbol
                    historical_data.append(stock_data)
            except Exception as e:
                st.warning(f"Could not download data for {symbol}: {e}")
    return pd.concat(historical_data) if historical_data else pd.DataFrame()

def display_predictions(predictions):
    """Visualize predicted returns."""
    st.subheader("Predicted Stock Returns")
    fig = go.Figure(data=[
        go.Bar(name='Predicted Returns', x=predictions.index, y=predictions.values)
    ])
    fig.update_layout(title='Predicted Stock Returns', xaxis_title='Stock Symbol', yaxis_title='Predicted Return (%)')
    st.plotly_chart(fig)

def recommend_stocks(live_data, predictions, risk_percentage, money, top_n=5):
    """Recommend stocks based on predicted returns and risk."""
    recommended_stocks = []
    for symbol, data in live_data.items():
        try:
            if symbol in predictions.index:
                predicted_return = predictions.loc[symbol]
                # Simulated beta value
                data['beta'] = np.random.uniform(0.5, 2.0)  # Placeholder for actual beta
                # Filter stocks based on risk tolerance
                if (risk_percentage < 33 and data['beta'] < 1) or \
                   (33 <= risk_percentage < 66 and 1 <= data['beta'] < 1.5) or \
                   (risk_percentage >= 66 and data['beta'] >= 1.5):
                    recommended_stocks.append((symbol, data, predicted_return))
        except Exception as e:
            st.warning(f"Error processing {symbol}: {e}")
    return sorted(recommended_stocks, key=lambda x: x[2], reverse=True)[:top_n]

def display_recommendations(recommended_stocks, money):
    """Display recommended stocks and investment breakdown."""
    st.subheader("Recommended Stocks:")
    investment_distribution = {}
    for symbol, data, predicted_return in recommended_stocks:
        investment = (predicted_return / 100) * money
        investment_distribution[symbol] = investment
        st.write(f"Symbol: {symbol}, Volatility (Beta): {data.get('beta', 'N/A')}, Predicted Returns: {predicted_return:.2f}%, Recommended Investment: ${investment:.2f}")
    st.write(f"Investment Distribution: {investment_distribution}")

def predict_stock_investment(model, live_data, le):
    """Prepare and predict stock investments using the model."""
    prepared_data = []
    for symbol, data in live_data.items():
        try:
            prepared_data.append({
                'symbol': symbol,
                'Open': data['open'],
                'High': data['dayHigh'],
                'Low': data['dayLow'],
                'Close': data['previousClose'],
                'Volume': data['volume']
            })
        except Exception as e:
            st.warning(f"Could not process data for {symbol}: {e}")
            continue
    # DataFrame preparation
    prepared_df = pd.DataFrame(prepared_data)
    current_date = pd.to_datetime('now')
    prepared_df['year'], prepared_df['month'], prepared_df['day'] = current_date.year, current_date.month, current_date.day
    le.classes_ = np.append(le.classes_, prepared_df['symbol'].unique())
    prepared_df['symbol_encoded'] = le.transform(prepared_df['symbol'])
    
    # Predict using model
    features = prepared_df[['symbol_encoded', 'year', 'month', 'day']]
    return pd.Series(model.predict(features), index=prepared_df['symbol'])

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
            historical_data['year'] = historical_data.index.year
            historical_data['month'] = historical_data.index.month
            historical_data['day'] = historical_data.index.day

            # Feature Engineering
            feature_columns = [col for col in historical_data.columns if col not in ['Symbol', 'year', 'month', 'day'] and historical_data[col].dtype in [np.float64, np.int64]]
            le = LabelEncoder()
            historical_data['symbol_encoded'] = le.fit_transform(historical_data['Symbol'])

            # Train-Test Split
            features = historical_data[['symbol_encoded', 'year', 'month', 'day'] + feature_columns]
            target = historical_data['Close']
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
            
            # Model Training
            rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_regressor.fit(X_train, y_train)
            
            # Evaluation
            y_pred = rf_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"Mean Squared Error on Test Set: {mse}")

            # Simulated live data
            live_data_sp500 = {symbol: {'open': 100, 'dayHigh': 105, 'dayLow': 95, 'previousClose': 100, 'volume': 1000} for symbol in tickers['Symbol']}
            predicted_returns = predict_stock_investment(rf_regressor, live_data_sp500, le)
            display_predictions(predicted_returns)

            # Recommend stocks based on user input
            recommended_stocks = recommend_stocks(live_data_sp500, predicted_returns, risk_percentage, money)
            display_recommendations(recommended_stocks, money)

            # Adjust Investment Distribution
            adjusted_distribution = adjust_investment_distribution(recommended_stocks, money)
            risk_reward_analysis(recommended_stocks)

            # Simulate Investment Scenarios
            time_horizon = st.slider('Set time horizon for simulation (weeks):', min_value=1, max_value=52, value=4)
            scenario_df = investment_scenarios(recommended_stocks, money, time_horizon)

            # Display Portfolio Performance Summary
            display_portfolio_performance(scenario_df)

def adjust_investment_distribution(recommended_stocks, money):
    """Allow users to adjust their investment allocation dynamically."""
    st.subheader("Adjust Investment Allocation")
    investment_distribution = {}
    for symbol, data, predicted_return in recommended_stocks:
        allocation_percentage = st.slider(f"Set allocation percentage for {symbol}:", 0, 100, value=20)
        investment = (allocation_percentage / 100) * money
        investment_distribution[symbol] = investment
        st.write(f"Allocated Investment: ${investment:.2f}")
        
    # Display investment distribution pie chart
    if investment_distribution:
        fig = go.Figure(data=[go.Pie(labels=list(investment_distribution.keys()), values=list(investment_distribution.values()))])
        fig.update_layout(title='Investment Distribution by Stock')
        st.plotly_chart(fig)
        
    return investment_distribution

def risk_reward_analysis(recommended_stocks):
    """Analyze and visualize risk vs reward for recommended stocks."""
    st.subheader("Risk vs Reward Analysis")
    risk_data = [{'symbol': symbol, 'predicted_return': predicted_return, 'beta': data.get('beta', 'N/A')} for symbol, data, predicted_return in recommended_stocks]
    risk_df = pd.DataFrame(risk_data)

    fig = go.Figure(data=go.Scatter(
        x=risk_df['beta'], 
        y=risk_df['predicted_return'],
        mode='markers',
        text=risk_df['symbol'],
        marker=dict(size=12, opacity=0.8)
    ))
    fig.update_layout(title='Risk vs Reward', xaxis_title='Beta (Risk)', yaxis_title='Predicted Return (%)')
    st.plotly_chart(fig)

def investment_scenarios(recommended_stocks, money, time_horizon):
    """Simulate investment scenarios based on user-defined time horizon."""
    scenario_df = pd.DataFrame(columns=['Stock', 'Time (Weeks)', 'Investment', 'Projected Return'])
    for symbol, data, predicted_return in recommended_stocks:
        projected_return = money * (predicted_return / 100) * (time_horizon / 52)
        scenario_df = scenario_df.append({'Stock': symbol, 'Time (Weeks)': time_horizon, 'Investment': money, 'Projected Return': projected_return}, ignore_index=True)
    return scenario_df

def display_portfolio_performance(scenario_df):
    """Display the performance summary of the user's portfolio."""
    st.subheader("Portfolio Performance Summary")
    st.write(scenario_df)
    
    total_investment = scenario_df['Investment'].sum()
    total_return = scenario_df['Projected Return'].sum()
    st.write(f"Total Investment: ${total_investment:.2f}")
    st.write(f"Total Projected Return: ${total_return:.2f}")

