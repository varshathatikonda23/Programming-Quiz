# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:21:02 2024

@author: thati
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import matplotlib.pyplot as plt

# Load and preprocess dataset
# Replace 'data.csv' with your dataset
data = pd.read_csv('data.csv', parse_dates=['date'], index_col='date')
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Split dataset into training and test sets
train_size = int(len(data_normalized) * 0.8)
train_data, test_data = data_normalized[:train_size], data_normalized[train_size:]

# Prepare data for LSTM
def create_dataset(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 10  # Number of time steps to look back
X_train, y_train = create_dataset(train_data, time_steps)
X_test, y_test = create_dataset(test_data, time_steps)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Define LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
loss = model.evaluate(X_test, y_test)

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Visualize predictions
plt.plot(scaler.inverse_transform(data_normalized), label='True Data')
plt.plot(range(train_size + time_steps, len(data_normalized)), predictions, label='Predictions')
plt.legend()
plt.show()

# Discussion and analysis
# Discuss your findings, challenges, and optimization steps here
