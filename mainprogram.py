import streamlit as st
import pandas as pd
import yfinance as yf
from yahoo_fin import stock_info
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from tqdm import tqdm

# Title of the app
st.title('Stock Predictor App')

# User input for investment details
money = st.number_input('Enter the amount of money:', min_value=0.0, value=1000.0)
time = st.number_input('Enter the time in weeks:', min_value=1, value=4)
risk_percentage = st.number_input('Enter risk percentage (0-100):', min_value=0.0, max_value=100.0, value=50.0)
returns = st.number_input('Enter expected returns (1-100):', min_value=1.0, max_value=100.0, value=10.0)

# Validate risk percentage input
if not (0 <= risk_percentage <= 100):
    st.error('Invalid input for risk percentage. Please enter a value between 0 and 100.')

# Button to fetch data and make predictions
if st.button('Predict Stocks'):
    # Fetch the list of S&P 500 companies
    tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

    # Function to get live data for companies
    def get_live_data_for_companies(tickers):
        live_data_for_companies = {}
        for symbol in tqdm(tickers['Symbol']):
            try:
                live_data = yf.Ticker(symbol).info
                live_data_for_companies[symbol] = live_data
            except Exception as e:
                st.warning(f"Could not fetch data for {symbol}: {e}")
        return live_data_for_companies

    # Prepare historical data for model training
    def prepare_data(tickers):
        historical_data = []
        for symbol in tqdm(tickers['Symbol']):
            try:
                stock_data = yf.download(symbol, period='3mo')
                stock_data['Symbol'] = symbol  
                historical_data.append(stock_data)
            except Exception as e:
                st.warning(f"Could not download data for {symbol}: {e}")
        historical_df = pd.concat(historical_data) if historical_data else pd.DataFrame()
        st.write("Historical Data Shape:", historical_df.shape)  # Debugging line
        return historical_df

    historical_data = prepare_data(tickers)


    # Feature engineering
    historical_data['year'] = historical_data.index.year
    historical_data['month'] = historical_data.index.month
    historical_data['day'] = historical_data.index.day
    historical_data.dropna(inplace=True)

    # Label encode the stock symbols
    le = LabelEncoder()
    historical_data['symbol_encoded'] = le.fit_transform(historical_data['Symbol'])

    # Define features and target variable
    features = historical_data[['symbol_encoded', 'year', 'month', 'day', 'Open', 'High', 'Low', 'Close', 'Volume']]
    target = historical_data['Adj Close']
    if historical_data.empty:
        st.error("No historical data retrieved. Please check the ticker symbols or your internet connection.")
    else:
    # Feature engineering
        historical_data['year'] = historical_data.index.year
        historical_data['month'] = historical_data.index.month
        historical_data['day'] = historical_data.index.day
        historical_data.dropna(inplace=True)

    # Label encode the stock symbols
    le = LabelEncoder()
    historical_data['symbol_encoded'] = le.fit_transform(historical_data['Symbol'])

    # Define features and target variable
    features = historical_data[['symbol_encoded', 'year', 'month', 'day', 'Open', 'High', 'Low', 'Close', 'Volume']]
    target = historical_data['Adj Close']

    if features.empty or target.empty:
        st.error("Feature or target data is empty. Check the historical data for issues.")
    else:
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Create a Random Forest Regressor
    rf_regressor = RandomForestRegressor()

    # Define hyperparameters for randomized search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Perform randomized search for hyperparameter tuning
    random_search = RandomizedSearchCV(rf_regressor, param_distributions=param_grid, n_iter=50, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)

    # Train the final model using the best parameters found
    final_rf_regressor = RandomForestRegressor(**random_search.best_params_)
    final_rf_regressor.fit(X_train, y_train)

    # Predict on the test set and calculate mean squared error
    y_pred = final_rf_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    st.write("Mean Squared Error on Test Set: ", mse)

    # Fetch live data
    live_data_sp500 = get_live_data_for_companies(tickers)

    # Function to predict stock investment returns
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
                st.warning(f"Error processing data for {symbol}: {e}")
        prepared_df = pd.DataFrame(prepared_data)
        prepared_df['year'] = pd.to_datetime('now').year
        prepared_df['month'] = pd.to_datetime('now').month
        prepared_df['day'] = pd.to_datetime('now').day
        prepared_df['symbol_encoded'] = le.transform(prepared_df['symbol'])
        features = prepared_df[['symbol_encoded', 'year', 'month', 'day', 'Open', 'High', 'Low', 'Close', 'Volume']]
        predicted_returns = model.predict(features)
        prepared_df['predicted_returns'] = predicted_returns
        prepared_df.set_index('symbol', inplace=True)
        return prepared_df['predicted_returns']

    # Predict returns using the model
    predicted_returns = predict_stock_investment(final_rf_regressor, live_data_sp500, le)
    st.write(predicted_returns)

    # Recommend stocks based on predicted returns and risk percentage
    def recommend_stocks(live_data_sp500, predicted_returns, risk_percentage, money, top_n=5):
        recommended_stocks = []
        for symbol, data in live_data_sp500.items():
            try:
                if symbol in predicted_returns.index:
                    predicted_return = predicted_returns.loc[symbol]
                    if risk_percentage < 33 and data['beta'] < 1:
                        recommended_stocks.append((symbol, data, predicted_return))
                    elif 33 <= risk_percentage < 66 and 1 <= data['beta'] < 1.5:
                        recommended_stocks.append((symbol, data, predicted_return))
                    elif risk_percentage >= 66 and data['beta'] >= 1.5:
                        recommended_stocks.append((symbol, data, predicted_return))
            except Exception as e:
                st.warning(f"Error processing {symbol}: {e}")
                pass
        recommended_stocks.sort(key=lambda x: x[2], reverse=True)
        return recommended_stocks[:top_n]

    # Get recommended stocks
    recommended_stocks = recommend_stocks(live_data_sp500, predicted_returns, risk_percentage, money, top_n=5)

    # Display recommended stocks with options to select
    st.write('### Recommended Stocks:')
    stock_options = st.multiselect("Select stocks to view details:", [symbol for symbol, _, _ in recommended_stocks])

    for symbol, data, returns in recommended_stocks:
        if symbol in stock_options:
            st.write(f"**Symbol:** {symbol}")
            st.write(f"**Company:** {data.get('longName', 'N/A')}")
            st.write(f"**Volatility (Beta):** {data.get('beta', 'N/A')}")
            st.write(f"**Market Cap:** {data.get('marketCap', 'N/A')}")
            st.write(f"**Industry:** {data.get('industry', 'N/A')}")
            st.write(f"**Predicted Returns:** {returns:.2f}")
            st.write(f"**Reason:** {'Low risk' if risk_percentage < 33 else 'Medium risk' if risk_percentage < 66 else 'High risk'} stock with appropriate beta value.")

            # Fetch news articles for the selected stock
            news_articles = fetch_news_articles(symbol)
            st.write(f'### News for {symbol}:')
            if news_articles:
                for article in news_articles:
                    st.write(f"**Title:** {article['title']}\n**Link:** [Read more]({article['link']})\n")
            else:
                st.write("No news articles found.")

    # Allow users to download recommended stocks as CSV
    if st.button('Download Recommended Stocks'):
        df_recommended = pd.DataFrame(recommended_stocks, columns=['Symbol', 'Data', 'Predicted Returns'])
        df_recommended['Market Cap'] = df_recommended['Data'].apply(lambda x: x.get('marketCap', 'N/A'))
        df_recommended['Industry'] = df_recommended['Data'].apply(lambda x: x.get('industry', 'N/A'))
        df_recommended.drop(columns=['Data'], inplace=True)
        df_recommended.to_csv('recommended_stocks.csv', index=False)
        st.success('Recommended stocks have been downloaded!')

    # Plot predicted returns using Plotly
    def plot_predicted_returns(recommended_stocks, model, live_data_sp500):
        predictions = []
        for symbol, data, returns in recommended_stocks:
            if symbol in live_data_sp500:
                input_data = pd.DataFrame([live_data_sp500[symbol]])
                input_data = input_data[['Close', 'High', 'Low', 'Open', 'Volume']]
                predicted_returns = model.predict(input_data)
                predictions.append({'Stock Symbol': symbol, 'Predicted Returns': predicted_returns[0]})
        
        predictions_df = pd.DataFrame(predictions)
        fig = px.bar(predictions_df, x='Stock Symbol', y='Predicted Returns', title='Predicted Returns for Recommended Stocks', color='Predicted Returns', color_continuous_scale='Viridis')
        st.plotly_chart(fig)

    plot_predicted_returns(recommended_stocks, final_rf_regressor, live_data_sp500)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to perform Monte Carlo simulations
def monte_carlo_simulation(stock_price, mu, sigma, num_simulations=1000, time_horizon=30):
    simulation_results = np.zeros((time_horizon, num_simulations))
    
    for i in range(num_simulations):
        daily_returns = np.random.normal(mu, sigma, time_horizon)
        price_paths = stock_price * np.exp(np.cumsum(daily_returns))
        simulation_results[:, i] = price_paths
    
    return simulation_results

# Function to plot Monte Carlo simulations
def plot_monte_carlo_simulation(simulations, stock_symbol):
    plt.figure(figsize=(14, 7))
    plt.plot(simulations, color='blue', alpha=0.1)
    plt.title(f'Monte Carlo Simulations for {stock_symbol}')
    plt.xlabel('Days')
    plt.ylabel('Price')
    
    # Plot mean and confidence intervals
    plt.plot(np.mean(simulations, axis=1), color='red', label='Mean Price Path')
    plt.fill_between(range(simulations.shape[0]), 
                     np.percentile(simulations, 1, axis=1), 
                     np.percentile(simulations, 99, axis=1), 
                     color='gray', alpha=0.3, label='1-99% Confidence Interval')
    
    plt.legend()
    plt.grid()
    plt.show()

# Function to run Monte Carlo simulations for the selected stocks
def run_monte_carlo_for_selected_stocks(recommended_stocks, num_simulations=1000, time_horizon=30):
    for symbol, data, returns in recommended_stocks:
        try:
            # Fetch historical data for Monte Carlo simulation
            stock_data = yf.download(symbol, period='1y')
            last_price = stock_data['Adj Close'][-1]
            log_returns = np.log(stock_data['Adj Close'] / stock_data['Adj Close'].shift(1)).dropna()
            mu = log_returns.mean()
            sigma = log_returns.std()

            # Run simulations
            simulations = monte_carlo_simulation(last_price, mu, sigma, num_simulations, time_horizon)

            # Plot results
            plot_monte_carlo_simulation(simulations, symbol)

        except Exception as e:
            st.warning(f"Error in running Monte Carlo simulation for {symbol}: {e}")

# Include this block in your Streamlit app where you want to run the Monte Carlo simulations
if st.button('Run Monte Carlo Simulations'):
    run_monte_carlo_for_selected_stocks(recommended_stocks)
