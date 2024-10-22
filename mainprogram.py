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

# Global variables to hold fetched data and predictions
historical_data = pd.DataFrame()
predicted_returns = pd.Series()

# Sidebar for investment details
st.sidebar.header("Investment Inputs")
money = st.sidebar.number_input('Enter the amount of money:', min_value=0.0, value=1000.0)
time = st.sidebar.number_input('Enter the time in weeks:', min_value=1, value=4)
risk_percentage = st.sidebar.number_input('Enter risk percentage (0-100):', min_value=0.0, max_value=100.0, value=50.0)
returns = st.sidebar.number_input('Enter expected returns (1-100):', min_value=1.0, max_value=100.0, value=10.0)

# Dropdown for historical data period
valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
historical_period = st.sidebar.selectbox('Select historical data period:', valid_periods)

# Function to get historical data for companies
def prepare_data(tickers, period):
    historical_data = []
    with st.spinner('Downloading historical data...'):
        for symbol in tqdm(tickers['Symbol'], desc="Downloading historical data"):
            try:
                stock_data = yf.download(symbol, period=period)
                if not stock_data.empty:
                    stock_data['Symbol'] = symbol
                    historical_data.append(stock_data)
            except Exception as e:
                st.warning(f"Could not download data for {symbol}: {e}")
    if historical_data:
        return pd.concat(historical_data)
    else:
        st.error("No historical data was fetched. Please check the input or try again.")
        return pd.DataFrame()

# Function to display predictions
def display_predictions(predictions):
    # Plot the predicted returns
    st.subheader("Predicted Stock Returns")
    fig = go.Figure(data=[
        go.Bar(name='Predicted Returns', x=predictions.index, y=predictions.values)
    ])
    fig.update_layout(title='Predicted Stock Returns', xaxis_title='Stock Symbol', yaxis_title='Predicted Return (%)')
    st.plotly_chart(fig)

# Function to recommend stocks based on predicted returns
def recommend_stocks(live_data, predictions, risk_percentage, money, top_n=5):
    recommended_stocks = []
    for symbol, data in live_data.items():
        try:
            if symbol in predictions.index:
                predicted_return = predictions.loc[symbol]
                # Beta placeholder
                data['beta'] = np.random.uniform(0.5, 2.0)  # Randomly generated beta for illustration
                # Filter stocks based on risk tolerance
                if risk_percentage < 33 and data['beta'] < 1:
                    recommended_stocks.append((symbol, data, predicted_return))
                elif 33 <= risk_percentage < 66 and 1 <= data['beta'] < 1.5:
                    recommended_stocks.append((symbol, data, predicted_return))
                elif risk_percentage >= 66 and data['beta'] >= 1.5:
                    recommended_stocks.append((symbol, data, predicted_return))
        except Exception as e:
            st.warning(f"Error processing {symbol}: {e}")
    # Sort by predicted returns and return top N
    recommended_stocks.sort(key=lambda x: x[2], reverse=True)
    return recommended_stocks[:top_n]

# Function to display recommended stocks and investment breakdown
def display_recommendations(recommended_stocks, money):
    st.subheader("Recommended Stocks:")
    investment_distribution = {}
    for symbol, data, predicted_return in recommended_stocks:
        st.write(f"Symbol: {symbol}")
        st.write(f"Volatility (Beta): {data.get('beta', 'N/A')}")
        st.write(f"Predicted Returns: {predicted_return:.2f}%")
        investment = (predicted_return / 100) * money
        investment_distribution[symbol] = investment
        st.write(f"Recommended Investment: ${investment:.2f}")
    st.write(f"Investment Distribution: {investment_distribution}")

# Function to predict stock investment based on live data
def predict_stock_investment(model, live_data, le):
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
    # Convert to DataFrame
    prepared_df = pd.DataFrame(prepared_data)
    prepared_df['year'] = pd.to_datetime('now').year
    prepared_df['month'] = pd.to_datetime('now').month
    prepared_df['day'] = pd.to_datetime('now').day
    # Handle unseen labels
    le.classes_ = np.append(le.classes_, prepared_df['symbol'].unique())
    prepared_df['symbol_encoded'] = le.transform(prepared_df['symbol'])
    # Predict using model
    features = prepared_df[['symbol_encoded', 'year', 'month', 'day']]
    return pd.Series(model.predict(features), index=prepared_df['symbol'])

# Main section to fetch data, train model, and predict stocks
if st.button('Predict Stocks'):
    # Fetch S&P 500 companies
    try:
        tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    except Exception as e:
        st.error(f"Error fetching S&P 500 companies: {e}")

    if not tickers.empty:
        # Prepare historical data
        historical_data = prepare_data(tickers, historical_period)
        if not historical_data.empty:
            # Feature engineering
            historical_data['year'] = historical_data.index.year
            historical_data['month'] = historical_data.index.month
            historical_data['day'] = historical_data.index.day

            # Dynamic column handling
            feature_columns = [col for col in historical_data.columns if col not in ['Symbol', 'year', 'month', 'day']]
            feature_columns = [col for col in feature_columns if historical_data[col].dtype in [np.float64, np.int64]]
            
            if 'Close' in feature_columns:
                # Label encode the stock symbols
                le = LabelEncoder()
                historical_data['symbol_encoded'] = le.fit_transform(historical_data['Symbol'])

                # Define features and target variable
                features = historical_data[['symbol_encoded', 'year', 'month', 'day'] + feature_columns]
                target = historical_data['Close']

                # Split data into train and test sets
                X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
                
                # Train Random Forest Regressor
                rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_regressor.fit(X_train, y_train)
                
                # Predict on the test set
                y_pred = rf_regressor.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                st.write(f"Mean Squared Error on Test Set: {mse}")

                # Predict stock returns using live data (placeholder for now)
                live_data_sp500 = {symbol: {'open': 100, 'dayHigh': 105, 'dayLow': 95, 'previousClose': 100, 'volume': 1000}
                                   for symbol in tickers['Symbol']}  # Placeholder

                predicted_returns = predict_stock_investment(rf_regressor, live_data_sp500, le)
                display_predictions(predicted_returns)

                # Recommend stocks based on user input
                recommended_stocks = recommend_stocks(live_data_sp500, predicted_returns, risk_percentage, money)
                display_recommendations(recommended_stocks, money)
                # Function to adjust investment allocation dynamically based on user preferences
                def adjust_investment_distribution(recommended_stocks, money):
                    st.subheader("Adjust Investment Allocation")
                    investment_distribution = {}
                    for symbol, data, predicted_return in recommended_stocks:
                        st.write(f"\nSymbol: {symbol}")
                        st.write(f"Volatility (Beta): {data.get('beta', 'N/A')}")
                        st.write(f"Predicted Returns: {predicted_return:.2f}%")
                        
                        # Let users adjust the allocation percentage for each stock
                        allocation_percentage = st.slider(f"Set allocation percentage for {symbol}:", 0, 100, 20)
                        
                        # Calculate the corresponding investment based on percentage
                        investment = (allocation_percentage / 100) * money
                        investment_distribution[symbol] = investment
                        st.write(f"Allocated Investment: ${investment:.2f}")
                    
                    # Plot the final investment distribution
                    st.write(f"\nFinal Investment Distribution:")
                    if investment_distribution:
                        fig = go.Figure(data=[
                            go.Pie(labels=list(investment_distribution.keys()), values=list(investment_distribution.values()))
                        ])
                        fig.update_layout(title='Investment Distribution by Stock')
                        st.plotly_chart(fig)
                    
                    return investment_distribution

                # Function to display risk/reward analysis
                def risk_reward_analysis(recommended_stocks):
                    st.subheader("Risk vs Reward Analysis")
                    risk_data = []
                    for symbol, data, predicted_return in recommended_stocks:
                        risk_data.append({
                            'symbol': symbol,
                            'predicted_return': predicted_return,
                            'beta': data.get('beta', 'N/A')
                        })
                    
                    # Convert to DataFrame for easier handling
                    risk_df = pd.DataFrame(risk_data)
                    
                    # Plot a scatter plot for risk vs reward
                    fig = go.Figure(data=go.Scatter(
                        x=risk_df['beta'], y=risk_df['predicted_return'], mode='markers', text=risk_df['symbol'],
                        marker=dict(size=10, color=risk_df['predicted_return'], colorscale='Viridis', showscale=True)
                    ))
                    fig.update_layout(title='Risk vs Reward (Beta vs Predicted Return)',
                                      xaxis_title='Volatility (Beta)', yaxis_title='Predicted Return (%)')
                    st.plotly_chart(fig)

                # Function to simulate different investment scenarios
                def investment_scenarios(recommended_stocks, money, time_horizon):
                    st.subheader("Simulate Investment Scenarios")
                    scenario_data = []
                    for symbol, data, predicted_return in recommended_stocks:
                        st.write(f"\nSimulating for {symbol}...")
                        scenario_return = predicted_return / 100 * money
                        final_value = money + scenario_return * time_horizon
                        scenario_data.append({
                            'symbol': symbol,
                            'initial_investment': money,
                            'predicted_return': predicted_return,
                            'time_horizon': time_horizon,
                            'final_value': final_value
                        })
                        st.write(f"Projected Final Value for {symbol}: ${final_value:.2f} after {time_horizon} weeks")
                    
                    return pd.DataFrame(scenario_data)

                # Function to display portfolio performance summary
                def display_portfolio_performance(scenario_df):
                    st.subheader("Portfolio Performance Summary")
                    if not scenario_df.empty:
                        # Display the scenario data as a table
                        st.write(scenario_df)
                        
                        # Plot the final values of each stock in the portfolio
                        fig = go.Figure(data=[
                            go.Bar(name='Final Value', x=scenario_df['symbol'], y=scenario_df['final_value'])
                        ])
                        fig.update_layout(title='Portfolio Final Values After Time Horizon',
                                          xaxis_title='Stock Symbol', yaxis_title='Final Value ($)')
                        st.plotly_chart(fig)

                # Call additional interactivity functions
                adjusted_distribution = adjust_investment_distribution(recommended_stocks, money)
                risk_reward_analysis(recommended_stocks)
                
                # Simulate different investment scenarios
                time_horizon = st.slider('Set time horizon for simulation (weeks):', min_value=1, max_value=52, value=4)
                scenario_df = investment_scenarios(recommended_stocks, money, time_horizon)
                
                # Display portfolio performance summary
                display_portfolio_performance(scenario_df)
