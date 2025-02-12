from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler

# Initialize Flask app
app = Flask(__name__)

# Load model & scaler
model = load_model("model.h5", custom_objects={"mse": MeanSquaredError()})
scaler = MinMaxScaler()
scaler.min_ = np.load("scaler.npy")
scaler.scale_ = np.load("scale_factor.npy")

# Load dataset
df = pd.read_csv("GoogleStock.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Function to predict stock prices
def predict_stock_prices(days):
    last_60_days = df[['Close']].values[-60:]
    last_60_days_scaled = scaler.transform(last_60_days)
    last_60_days_scaled = np.reshape(last_60_days_scaled, (1, 60, 1))

    # Predict future prices
    future_predictions = model.predict(last_60_days_scaled)
    future_predictions = scaler.inverse_transform(future_predictions[:, :days])

    # Generate future dates dynamically
    last_date = df.index[-1]
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, days + 1)]

    # Store predictions in DataFrame
    predicted_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions[0]})
    
    return predicted_df

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        days = int(request.form['days'])
        predictions = predict_stock_prices(days)
        return jsonify(predictions.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)})

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
