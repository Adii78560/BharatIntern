import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

ticker = 'AAPL'
start_date = '2010-01-01'
end_date = '2024-01-01'

data = yf.download(ticker, start=start_date, end=end_date)

df = data[['Close']]

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)


train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# def create_sequences(data, seq_length):
#     sequences = []
#     for i in range(len(data) - seq_length):
#         X = data[i:i+seq_length]
#         y = data[i+seq_length]
#         sequences.append((X, y))
#     return np.array(sequences)


def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 30  
train_X, train_y = create_sequences(train_data, seq_length)
test_X, test_y = create_sequences(test_data, seq_length)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(train_X, train_y, batch_size=64, epochs=100)


test_loss = model.evaluate(test_X, test_y)
print(f'Test Loss: {test_loss}')


predictions = model.predict(test_X)


predictions = scaler.inverse_transform(predictions)
test_data = scaler.inverse_transform(test_y)


plt.figure(figsize=(16, 8))
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(test_data, label='True Price', color='blue')
plt.plot(predictions, label='Predicted Price', color='red')
plt.legend()
plt.show()
