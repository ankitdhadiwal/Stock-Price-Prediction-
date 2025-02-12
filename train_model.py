import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model

# Load dataset
df = pd.read_csv("GoogleStock.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Prepare data
data = df[['Close']].values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Function to create sequences
def create_sequences(data, time_steps=60, forecast_horizon=7):
    X, y = [], []
    for i in range(len(data) - time_steps - forecast_horizon):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps: i + time_steps + forecast_horizon])
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, time_steps=60, forecast_horizon=7)

# Split dataset
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape input for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(7)  # Predicting 7 days
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_test, y_test))

# Save model and scaler
model.save("model.h5")
np.save("scaler.npy", scaler.min_)
np.save("scale_factor.npy", scaler.scale_)

print("✅ Model saved successfully!")
