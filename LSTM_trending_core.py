# LSTM_trending_core.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from datetime import timedelta, date, datetime
from vnstock import Vnstock
import tensorflow as tf
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

def calculate_indicators(df):
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df.dropna()

def create_dataset(dataset, time_step, predict_days=10):
    X, y = [], []
    for i in range(time_step, len(dataset) - predict_days + 1):
        X.append(dataset[i-time_step:i])
        y.append(dataset[i:i+predict_days, 0])
    return np.array(X), np.array(y)

def next_business_days(start_date, num_days):
    dates = []
    current_date = pd.to_datetime(start_date)
    while len(dates) < num_days:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:
            dates.append(current_date)
    return dates

def forecast_lstm(symbol, predict_days=10, time_step=150):
    vns = Vnstock()
    df = vns.stock(symbol=symbol, source='VCI').quote.history(
        start='2024-01-01', end=date.today().strftime('%Y-%m-%d'), interval='1D'
    )
    df = calculate_indicators(df)
    data = df[['close', 'volume', 'MACD', 'RSI']].copy()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = create_dataset(scaled_data, time_step, predict_days)
    if len(X) < 10:
        return None, None

    train_size = int(len(X) * 0.9)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = Sequential()
    model.add(LSTM(128, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(predict_days))
    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[early_stop], verbose=0)

    last_input = scaled_data[-time_step:]
    future_input = np.expand_dims(last_input, axis=0)
    future_pred = model.predict(future_input)
    future_pred_price = scaler.inverse_transform(
        np.hstack([future_pred[0].reshape(-1, 1), np.zeros((predict_days, 3))])
    )[:, 0]

    future_dates = next_business_days(df['time'].iloc[-1], predict_days)
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecasted Close Price (VND)': np.round(future_pred_price, 2)
    })

    # Chart (show 150 ngày gần nhất + 10 ngày dự báo)
    plt.figure(figsize=(12, 6))
    recent_mask = df['time'] >= (df['time'].iloc[-1] - pd.Timedelta(days=150))
    close_prices = df['close'][recent_mask].reset_index(drop=True)
    close_dates = df['time'][recent_mask].reset_index(drop=True)
    plt.plot(close_dates, close_prices, label='Giá đóng cửa thực tế', color='steelblue')
    forecast_x = future_dates
    forecast_y = future_pred_price
    plt.plot(forecast_x, forecast_y, label='Giá dự báo', color='orange', marker='o')
    plt.title(f'Dự báo giá đóng cửa {symbol} ({predict_days} ngày tới)')
    plt.xlabel('Ngày')
    plt.ylabel('Giá đóng cửa (VND)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig = plt.gcf()
    plt.close()
    return forecast_df, fig
