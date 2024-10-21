import streamlit as st
import pandas as pd
import yfinance as yf
from yahoo_fin import stock_info
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm

# Title of the app
st.title('Interactive Stock Predictor App')

# User input for investment details
money = st.number_input('Enter the amount of money:', min_value=0.0, value=1000.0)
time = st.number_input('Enter the time in weeks:', min_value=1, value=4)
risk_percentage = st.number_input('Enter risk percentage (0-100):', min_value=0.0, max_value=100.0, value=50.0)
returns = st.number_input('Enter expected returns (1-100):', min_value=1.0, max_value=100.0, value=10.0)

# User input for historical period using a dropdown menu
valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
historical_period = st.selectbox('Select historical data period:', valid_periods)

# Validate risk percentage input
if not (0 <= risk_percentage <= 100):
    st.error('Invalid input for risk percentage. Please enter a value between 0 and 100.')

# Button to fetch data and make predictions
if st.button('Predict Stocks'):
    # Fetch the list of S&P 500 companies
    try:
        tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    except Exception as e:
        st.error(f"Error fetching S&P 500 companies: {e}")
        tickers = pd.DataFrame()

        if not tickers.empty:
        # Function to get historical data for companies
        # Function to get historical data for companies
            def prepare_data(tickers, period):
                historical_data = []
                with st.spinner('Downloading historical data...'):
                    for symbol in tqdm(tickers['Symbol'], desc="Downloading historical data"):
                        try:
                            stock_data = yf.download(symbol, period=period)  # Use selected period directly
                            if not stock_data.empty:
                                stock_data['Symbol'] = symbol
                                historical_data.append(stock_data)
                        except Exception as e:
                              st.warning(f"Could not download data for {symbol}: {e}")

            if historical_data:
        # Concatenate all historical data into a single DataFrame at once
                return pd.concat(historical_data)
            else:
                st.error("No historical data was fetched. Please check the input or try again.")
                return pd.DataFrame()  # Return empty DataFrame

# Prepare historical data
historical_data = prepare_data(tickers, historical_period)

# Feature engineering
if not historical_data.empty:
    # Instead of adding columns one by one, collect data in a dictionary
    data_to_add = {
        'year': historical_data.index.year,
        'month': historical_data.index.month,
        'day': historical_data.index.day,
    }
    
    # Now create a new DataFrame with the additional columns
    additional_data_df = pd.DataFrame(data_to_add, index=historical_data.index)
    
    # Join the two DataFrames
    historical_data = pd.concat([historical_data, additional_data_df], axis=1)

    # Drop any rows with missing values
    historical_data.dropna(inplace=True)

    # Label encode the stock symbols
    le = LabelEncoder()
    historical_data['symbol_encoded'] = le.fit_transform(historical_data['Symbol'])

    # Define features and target variable
    features = historical_data[['symbol_encoded', 'year', 'month', 'day', 'Open', 'High', 'Low', 'Close', 'Volume']]
    target = historical_data['Adj Close']

    # Ensure that the features and target are valid
    if features.empty or target.empty:
        st.error("Feature or target data is empty. Check the historical data for issues.")

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
     with st.spinner('Training the model...'):
            random_search = RandomizedSearchCV(rf_regressor, param_distributions=param_grid, n_iter=50, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
            random_search.fit(X_train, y_train)

                # Train the final model using the best parameters found
            final_rf_regressor = RandomForestRegressor(**random_search.best_params_)
            final_rf_regressor.fit(X_train, y_train)

                # Predict on the test set and calculate mean squared error
             y_pred = final_rf_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

             st.write("Mean Squared Error on Test Set: ", mse)

                # Fetch live data for companies
            live_data_for_companies = {}
            for symbol in tqdm(tickers['Symbol'], desc="Fetching live data"):
                 try:
                    live_data = yf.Ticker(symbol).info
                    live_data_for_companies[symbol] = live_data
                 except Exception as e:
                    st.warning(f"Could not fetch data for {symbol}: {e}")

                # Function to predict stock investment returns
                def predict_stock_investment(model, live_data, le):
                    prepared_data = []
                    for symbol, data in live_data.items():
                        try:
                            if all(k in data for k in ['open', 'dayHigh', 'dayLow', 'previousClose']):
                                prepared_data.append({
                                    'symbol': symbol,
                                    'Open': data['open'],
                                    'High': data['dayHigh'],
                                    'Low': data['dayLow'],
                                    'Close': data['previousClose'],
                                    'Volume': data['volume']
                                })
                        except Exception as e:
                            st.warning(f"Missing data for {symbol}, skipping...")
                    
                    if not prepared_data:
                        st.error("No valid live data available for prediction.")
                        return pd.Series()  # Return an empty series or handle as necessary

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
                predicted_returns = predict_stock_investment(final_rf_regressor, live_data_for_companies, le)

                if not predicted_returns.empty:
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
                    recommended_stocks = recommend_stocks(live_data_for_companies, predicted_returns, risk_percentage, money)

                    with st.expander("Recommended Stocks Based on Predicted Returns"):
                        for stock in recommended_stocks:
                            st.write(f"Stock: {stock[0]}, Predicted Return: {stock[2]}, Beta: {stock[1].get('beta', 'N/A')}")
                
                    # Monte Carlo Simulation
                    def monte_carlo_simulation(predicted_returns, iterations=1000):
                        simulated_prices = []
                        for _ in range(iterations):
                            price_series = [money]
                            for _ in range(time * 5):  # Assuming 5 trading days per week
                                daily_return = np.random.normal(loc=predicted_returns.mean() / 100, scale=predicted_returns.std() / 100)
                                price_series.append(price_series[-1] * (1 + daily_return))
                            simulated_prices.append(price_series[-1])
                        return simulated_prices

                    simulated_prices = monte_carlo_simulation(predicted_returns)

                    # Plotting the simulation results
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=simulated_prices, name='Simulated Prices', opacity=0.75))
                    fig.add_trace(go.Scatter(x=[np.mean(simulated_prices)] * 2, y=[0, max(np.histogram(simulated_prices, bins=50)[0])], mode='lines', name='Mean Expected Price', line=dict(color='red', width=2)))

                    fig.update_layout(title='Monte Carlo Simulation Results', xaxis_title='Simulated Prices', yaxis_title='Frequency')
                    st.plotly_chart(fig)
                else:
                    st.warning("No predicted returns available for the stocks.")
        else:
            st.warning("No historical data available for analysis.")
