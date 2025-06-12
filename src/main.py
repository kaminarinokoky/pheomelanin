import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import requests
import talib

# Step 1: Load and Preprocess Data
def load_data(file_path):
    """
    Loads trade data from a CSV file and adds technical indicators.
    
    :param file_path: Path to the CSV file with trade data.
    :return: Processed DataFrame with features and labels.
    """
    df = pd.read_csv(file_path)
    
    # Ensure necessary columns exist
    if 'price' not in df.columns or 'time' not in df.columns:
        raise ValueError("CSV must contain 'price' and 'time' columns")
    
    # Convert time to datetime and sort
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    
    # Add technical indicators
    df['sma_20'] = talib.SMA(df['price'], timeperiod=20)
    df['rsi'] = talib.RSI(df['price'], timeperiod=14)
    
    # Create labels: 1 (buy) if next day's price is higher, 0 (sell) otherwise
    df['label'] = (df['price'].shift(-1) > df['price']).astype(int)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def prepare_data(df, sequence_length=60):
    """
    Prepares data for LSTM model by creating sequences and scaling.
    
    :param df: DataFrame with price, sma_20, rsi, and label columns.
    :param sequence_length: Number of time steps in each sequence.
    :return: X_train, X_test, y_train, y_test, scaler.
    """
    features = ['price', 'sma_20', 'rsi']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(df['label'].iloc[i])
    
    X, y = np.array(X), np.array(y)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler

# Step 2: Build LSTM Model
def build_model(sequence_length, n_features):
    """
    Builds an LSTM model for binary classification.
    
    :param sequence_length: Number of time steps in each sequence.
    :param n_features: Number of features (e.g., price, sma_20, rsi).
    :return: Compiled Keras model.
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, n_features)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 3: Train Model
def train_model(model, X_train, y_train, X_test, y_test):
    """
    Trains the LSTM model.
    
    :param model: Keras model.
    :param X_train, y_train: Training data and labels.
    :param X_test, y_test: Testing data and labels.
    :return: Trained model.
    """
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    return model

# Step 4: Backtest Model
def backtest_model(model, X_test, y_test, df, scaler, sequence_length):
    """
    Backtests the model by simulating trading.
    
    :param model: Trained Keras model.
    :param X_test, y_test: Testing data and labels.
    :param df: Original DataFrame with price data.
    :param scaler: Fitted MinMaxScaler.
    :param sequence_length: Number of time steps in each sequence.
    :return: Percentage return from backtesting.
    """
    predictions = model.predict(X_test)
    predictions = (predictions > 0.5).astype(int)
    
    initial_cash = 10000
    cash = initial_cash
    shares = 0
    for i in range(len(predictions)):
        price_idx = sequence_length + i
        if price_idx >= len(df):
            break
        price = df['price'].iloc[price_idx]
        if predictions[i] == 1 and cash > 0:  # Buy
            shares = cash / price
            cash = 0
        elif predictions[i] == 0 and shares > 0:  # Sell
            cash = shares * price
            shares = 0
    
    final_value = cash + shares * df['price'].iloc[-1]
    return (final_value - initial_cash) / initial_cash * 100

# Step 5: Placeholder for Trading with Nobitex API
def place_order(api_token, src_currency, dst_currency, order_type, amount, price):
    """
    Placeholder for placing a trade order via Nobitex API (requires authentication).
    
    :param api_token: Nobitex API token.
    :param src_currency: Source currency (e.g., "btc").
    :param dst_currency: Destination currency (e.g., "rls").
    :param order_type: "buy" or "sell".
    :param amount: Amount to trade.
    :param price: Price for the order.
    :return: Response from the API.
    """
    url = "https://api.nobitex.ir/market/orders/add"
    headers = {"Authorization": f"Token {api_token}"}
    data = {
        "type": order_type,
        "srcCurrency": src_currency,
        "dstCurrency": dst_currency,
        "amount": amount,
        "price": price
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to place order: {response.status_code}")

# Main Execution
if __name__ == "__main__":
    # Load and preprocess data
    file_path = "btc_irr_trades.csv"  # Generated by nobitex_data_fetcher.py
    try:
        df = load_data(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()
    
    # Prepare data for LSTM
    sequence_length = 60
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, sequence_length)
    
    # Build and train model
    model = build_model(sequence_length, len(['price', 'sma_20', 'rsi']))
    model = train_model(model, X_train, y_train, X_test, y_test)
    
    # Backtest
    return_percent = backtest_model(model, X_test, y_test, df, scaler, sequence_length)
    print(f"Backtest Return: {return_percent:.2f}%")
    
    # Save model for future use
    model.save("nobitex_lstm_model.h5")
    
    # Placeholder for live trading (uncomment and add your API token)
    # api_token = "your_nobitex_api_token"
    # try:
    #     order_response = place_order(api_token, "btc", "rls", "buy", 0.001, 500000000)
    #     print("Order Response:", order_response)
    # except Exception as e:
    #     print(f"Error placing order: {e}")