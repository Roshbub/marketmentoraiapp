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

# Button to fetch data and make predictions
if st.button('Predict Stocks'):
    # Initialize tickers to an empty DataFrame
    tickers = pd.DataFrame()  
    try:
        # Fetch the list of S&P 500 companies
        tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    except Exception as e:
        st.error(f"Error fetching S&P 500 companies: {e}")

    # Check if tickers DataFrame is not empty after try-except
    if not tickers.empty:
        # Prepare historical data
        historical_data = prepare_data(tickers, historical_period)

        # Check if historical data is empty
        if not historical_data.empty:
            # Feature engineering
            historical_data['year'] = historical_data.index.year
            historical_data['month'] = historical_data.index.month
            historical_data['day'] = historical_data.index.day
        print(historical_data.columns.tolist())

            # Dynamic column handling - only use available columns
            feature_columns = [col for col in historical_data.columns if col not in ['Symbol', 'year', 'month', 'day']]
            feature_columns = [col for col in feature_columns if historical_data[col].dtype in [np.float64, np.int64]]  # Use only numeric columns
            print(feature_columns.tolist())
            # Debug output: Display count of missing values in each column
            st.write("Missing values count:", historical_data.isnull().sum())
            
            # Ensure all required columns exist before dropping NaNs
            required_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in historical_data.columns]
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
            else:
           # Check if the DataFrame is empty after dropping NaNs
                if historical_data.empty:
                    st.error("DataFrame is empty after dropping NaNs. Please check the data.")
                else:
                    # Label encode the stock symbols
                    le = LabelEncoder()
                    historical_data['symbol_encoded'] = le.fit_transform(historical_data['Symbol'])

                    # Define features and target variable
                    features = historical_data[['symbol_encoded', 'year', 'month', 'day'] + feature_columns]
                    target = historical_data['Close']  # Assuming 'Close' will always be there for the target variable

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
                        features = prepared_df[['symbol_encoded', 'year', 'month', 'day'] + feature_columns]
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
                    st.write("Predicted Returns:", predicted_returns)

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

                        # Sort the recommended stocks by predicted returns in descending order and select the top N
                        recommended_stocks.sort(key=lambda x: x[2], reverse=True)
                        return recommended_stocks[:top_n]

                    # Recommend stocks based on predicted returns and risk percentage
                    recommended_stocks = recommend_stocks(live_data_sp500, predicted_returns, risk_percentage, money, top_n=5)

                    # Print recommended stocks and reasons for selection
                    st.write('\nRecommended Stocks:')
                    investment_distribution = {}
                    for symbol, data, returns in recommended_stocks:
                        st.write(f"\nSymbol: {symbol}")
                        st.write(f"Volatility (Beta): {data.get('beta', 'N/A')}")
                        st.write(f"Predicted Returns: {returns:.2f}%")
                        investment = (returns / 100) * money
                        investment_distribution[symbol] = investment
                        st.write(f"Recommended Investment: ${investment:.2f}")

                    st.write(f"\nInvestment Distribution: {investment_distribution}")

                    # Create a plot for predicted returns
                    fig = go.Figure(data=[
                        go.Bar(name='Predicted Returns', x=predicted_returns.index, y=predicted_returns.values)
                    ])
                    fig.update_layout(title='Predicted Stock Returns', xaxis_title='Stock Symbol', yaxis_title='Predicted Return (%)')
                    st.plotly_chart(fig)

                
