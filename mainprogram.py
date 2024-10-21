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
st.title('Interactive Stock Predictor App')

# User input for investment details
money = st.number_input('Enter the amount of money:', min_value=0.0, value=1000.0)
time = st.number_input('Enter the time in weeks:', min_value=1, value=4)
risk_percentage = st.number_input('Enter risk percentage (0-100):', min_value=0.0, max_value=100.0, value=50.0)
returns = st.number_input('Enter expected returns (1-100):', min_value=1.0, max_value=100.0, value=10.0)

# User input for historical period using a dropdown menu
valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
historical_period = st.selectbox('Select historical data period:', valid_periods)

# Create a DataFrame with the user input
user_data = pd.DataFrame({
    'money': [money],
    'time': [time],
    'risk_percentage': [risk_percentage],
    'returns': [returns]
})

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

        # Prepare historical data
        historical_data = prepare_data(tickers, historical_period)

        # Check if historical data is empty
        if historical_data.empty:
            st.error("No historical data available. Please try a different period or check the input.")
        else:
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

            # Ensure that features and target are valid
            if features.empty or target.empty:
                st.error("Features or target data is empty. Unable to proceed with model training.")
            else:
                # Split data into train and test sets
                X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

                # Create and train Random Forest Regressor
                rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_regressor.fit(X_train, y_train)

                # Predict on the test set and calculate mean squared error
                y_pred = rf_regressor.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                st.write("Mean Squared Error on Test Set: ", mse)
        # Function to predict stock investment
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

            # Handle unseen labels during transformation
            le.classes_ = np.append(le.classes_, prepared_df['symbol'].unique())
            prepared_df['symbol_encoded'] = le.transform(prepared_df['symbol'])

            # Define features for prediction
            features = prepared_df[['symbol_encoded', 'year', 'month', 'day', 'Open', 'High', 'Low', 'Close', 'Volume']]
            predicted_returns = model.predict(features)

            # Add the predicted returns to the DataFrame
            prepared_df['predicted_returns'] = predicted_returns
            prepared_df.set_index('symbol', inplace=True)

            return prepared_df['predicted_returns']

        # Assuming live_data_sp500 is available; otherwise, you need to define it
        live_data_sp500 = {symbol: {'open': 100, 'dayHigh': 105, 'dayLow': 95, 'previousClose': 100, 'volume': 1000}
                           for symbol in tickers['Symbol']}  # Placeholder for live data

        # Predict returns using the machine learning model
        predicted_returns = predict_stock_investment(rf_regressor, live_data_sp500, le)
        st.write(predicted_returns)

        # Function to recommend stocks based on predicted returns and risk percentage
        def recommend_stocks(live_data_sp500, predicted_returns, risk_percentage, money, top_n=5):
            recommended_stocks = []
            for symbol, data in live_data_sp500.items():
                try:
                    if symbol in predicted_returns.index:
                        predicted_return = predicted_returns.loc[symbol]
                        # Placeholder for beta; should be fetched from stock_info or a reliable source
                        data['beta'] = np.random.uniform(0.5, 2.0)  # Example: Randomly generated beta for illustration

                        if risk_percentage < 33 and data['beta'] < 1:
                            recommended_stocks.append((symbol, data, predicted_return))
                        elif 33 <= risk_percentage < 66 and 1 <= data['beta'] < 1.5:
                            recommended_stocks.append((symbol, data, predicted_return))
                        elif risk_percentage >= 66 and data['beta'] >= 1.5:
                            recommended_stocks.append((symbol, data, predicted_return))
                except Exception as e:
                    st.warning(f"Error processing {symbol}: {e}")
                    continue

                    # Recommend stocks based on predicted returns and risk percentage
            recommended_stocks = recommend_stocks(live_data_sp500, predicted_returns, risk_percentage, money, top_n=5)

            # Sort the recommended stocks by predicted returns in descending order and select the top N
            recommended_stocks.sort(key=lambda x: x[2], reverse=True)
            return recommended_stocks[:top_n]

        # Print recommended stocks and reasons for selection
        st.write('\nRecommended Stocks:')
        for symbol, data, returns in recommended_stocks:
            st.write(f"\nSymbol: {symbol}")
            st.write(f"Company: {data.get('longName', 'N/A')}")
            st.write(f"Volatility (Beta): {data.get('beta', 'N/A')}")
            st.write(f"Predicted Return: {returns:.2f}")

            # Monte Carlo simulations
            def monte_carlo_simulation(symbol, initial_price, predicted_return, volatility, days=30, simulations=1000):
                price_paths = np.zeros((days, simulations))
                price_paths[0] = initial_price
                
                for t in range(1, days):
                    random_returns = np.random.normal(predicted_return / 100, volatility / 100, simulations)
                    price_paths[t] = price_paths[t - 1] * (1 + random_returns)
                
                return price_paths

            # Assuming a volatility placeholder (standard deviation of historical returns)
            historical_volatility = np.random.uniform(0.1, 0.3)  # Placeholder volatility
            initial_price = live_data_sp500[symbol]['previousClose']  # Placeholder for actual price
            simulations = monte_carlo_simulation(symbol, initial_price, returns, historical_volatility)

            # Calculate the average price path (best-fit line)
            average_path = simulations.mean(axis=1)

            # Calculate the Sharpe ratio
            average_return = average_path[-1] - initial_price
            risk_free_rate = 0.01  # Example risk-free rate
            sharpe_ratio = (average_return - risk_free_rate) / (historical_volatility * np.sqrt(252))  # Annualized Sharpe Ratio

            # Plotting the results using Plotly
            fig = go.Figure()
            for i in range(simulations.shape[1]):
                fig.add_trace(go.Scatter(x=np.arange(0, simulations.shape[0]), y=simulations[:, i], mode='lines', line=dict(color='blue', width=0.5), showlegend=False))

            fig.add_trace(go.Scatter(x=np.arange(0, len(average_path)), y=average_path, mode='lines', line=dict(color='red', width=2), name='Average Path'))
            fig.update_layout(title=f'Monte Carlo Simulations for {symbol}', xaxis_title='Days', yaxis_title='Price', showlegend=True)
            st.plotly_chart(fig)

            st.write(f"Sharpe Ratio for {symbol}: {sharpe_ratio:.2f}")

