# Import packages
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
from textblob import TextBlob
import seaborn as sns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from statsmodels.tsa.stattools import adfuller
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
            except:
                st.warning(f"Could not fetch data for {symbol}")
        return live_data_for_companies

    # Prepare historical data for model training
    def prepare_data(tickers):
        historical_data = []
        for symbol in tqdm(tickers['Symbol']):
            try:
                stock_data = yf.download(symbol, period='3mo')
                stock_data['Symbol'] = symbol
                historical_data.append(stock_data)
            except:
                st.warning(f"Could not download data for {symbol}")
        return pd.concat(historical_data)

    historical_data = prepare_data(tickers)

    # Feature engineering
    historical_data['year'] = historical_data.index.year
    historical_data['month'] = historical_data.index.month
    historical_data['day'] = historical_data.index.day
    historical_data = historical_data.dropna()

    # Label encode the stock symbols
    le = LabelEncoder()
    historical_data['symbol_encoded'] = le.fit_transform(historical_data['Symbol'])

    # Define features and target variable
    features = historical_data[['symbol_encoded', 'year', 'month', 'day', 'Open', 'High', 'Low', 'Close', 'Volume']]
    target = historical_data['Adj Close']

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
            except:
                pass
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

    # Display recommended stocks
    st.write('### Recommended Stocks:')
    for symbol, data, returns in recommended_stocks:
        st.write(f"**Symbol:** {symbol}")
        st.write(f"**Company:** {data['longName']}")
        st.write(f"**Volatility (Beta):** {data['beta']}")
        st.write(f"**Market Cap:** {data['marketCap']}")
        st.write(f"**Industry:** {data['industry']}")
        st.write(f"**Predicted Returns:** {returns}")
        st.write(f"**Reason:** {'Low risk' if risk_percentage < 33 else 'Medium risk' if risk_percentage < 66 else 'High risk'} stock with appropriate beta value.")

    # Fetch news articles for recommended stocks
    def fetch_news_articles(symbol):
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            if not news:
                st.warning(f"No news articles found for {symbol}")
            return news[:5]  # Return the top 5 news articles
        except Exception as e:
            st.warning(f"Error fetching news for {symbol}: {e}")
            return []

    # Fetch and display news articles
    for stock, data, returns in recommended_stocks:
        news_articles = fetch_news_articles(stock)
        st.write(f'### News for {stock}:')
        if news_articles:
            for article in news_articles:
                st.write(f"**Title:** {article['title']}\n**Link:** {article['link']}\n")
        else:
            st.write("No news articles found.")

    # Plot predicted returns
    def plot_predicted_returns(recommended_stocks, model, live_data_sp500, le):
        plt.figure(figsize=(14, 7))
        plotted = False
        for symbol, data, returns in recommended_stocks:
            try:
                if symbol in live_data_sp500:
                    input_data = pd.DataFrame([live_data_sp500[symbol]])
                    input_data = input_data[['Close', 'High', 'Low', 'Open', 'Volume']]
                    predicted_returns = model.predict(input_data)
                    plt.plot(predicted_returns, label=symbol)
                    plotted = True
            except Exception as e:
                st.warning(f"Error plotting for {symbol}: {e}")
                continue
        plt.title('Predicted Returns for Recommended Stocks')
        plt.xlabel('Stock Symbols')
        plt.ylabel('Predicted Returns')
        plt.legend()
        plt.grid()
        if plotted:
            st.pyplot()

    # Plot cumulative returns for recommended stocks
    def plot_cumulative_returns(recommended_stocks):
        plt.figure(figsize=(14, 7))
        for symbol, data, returns in recommended_stocks:
            cumulative_return = (1 + returns) ** (1 / time) - 1
            plt.plot(cumulative_return, label=symbol)
        plt.title('Cumulative Returns for Recommended Stocks')
        plt.xlabel('Time (weeks)')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid()
        st.pyplot()

    # Calculate Sharpe ratios for recommended stocks
    def calculate_sharpe_ratio(recommended_stocks):
        sharpe_ratios = {}
        for symbol, data, returns in recommended_stocks:
            risk_free_rate = 0.01  # Assume a constant risk-free rate
            sharpe_ratio = (returns - risk_free_rate) / np.std(returns)
            sharpe_ratios[symbol] = sharpe_ratio
        return sharpe_ratios

    # Perform Monte Carlo simulation for portfolio returns
    def monte_carlo_simulation(recommended_stocks, num_simulations=1000):
        results = []
        for _ in range(num_simulations):
            weights = np.random.rand(len(recommended_stocks))
            weights /= np.sum(weights)  # Normalize to sum to 1
            portfolio_return = np.sum([data[2] * weight for data, weight in zip(recommended_stocks, weights)])
            results.append(portfolio_return)
        return results

    # Run and plot Monte Carlo simulation results
    st.subheader('Monte Carlo Simulation Results')
    simulated_portfolios = monte_carlo_simulation(recommended_stocks)
    plt.figure(figsize=(14, 7))
    plt.hist(simulated_portfolios, bins=50, alpha=0.7)
    plt.title('Monte Carlo Simulation of Portfolio Returns')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    st.pyplot()
