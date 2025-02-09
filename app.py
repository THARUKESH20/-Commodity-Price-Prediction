from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Load the dataset
df = pd.read_excel('combined.xlsx')

# Fill missing values with median of each column
columns_to_fill = ['Rice', 'Wheat', 'Atta', 'Gramdal', 'Turdal', 'Uraddal', 'Moongdal', 'Masoordal', 
                    'Sugar', 'Milk', 'Groundnutoil', 'Mustardoil', 'Vanaspati', 'Soyaoil', 'Sunfloweroil', 
                    'Palmoil', 'Gur', 'Tealoose', 'Saltpack', 'Potato', 'Onion', 'Tomato']
for column in columns_to_fill:
    median_value = df[column].median()
    df[column].fillna(median_value, inplace=True)

# Encode categorical data
label_encode = LabelEncoder()
df['State'] = label_encode.fit_transform(df['State'])

# List of relevant features (commodities)
features = [
    'Rice', 'Wheat', 'Atta', 'Gramdal', 'Turdal', 'Uraddal', 'Moongdal', 'Masoordal',
    'Sugar', 'Milk', 'Groundnutoil', 'Mustardoil', 'Vanaspati', 'Soyaoil', 'Sunfloweroil',
    'Palmoil', 'Gur', 'Tealoose', 'Saltpack', 'Potato', 'Onion', 'Tomato'
]

# Function to train ARIMA model
def train_arima_model(data, order=(5, 1, 0)):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit

# Function to predict future prices using ARIMA model
def predict_future_price_arima(model_fit, steps=7):
    forecast = model_fit.forecast(steps=steps)
    return forecast

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_state_name = request.form['state']
    input_feature = request.form['feature']

    if input_state_name not in label_encode.classes_:
        return render_template('index.html', prediction=None, error=f"State '{input_state_name}' not found.")

    input_state = label_encode.transform([input_state_name])[0]

    if input_feature not in features:
        return render_template('index.html', prediction=None, error=f"Feature '{input_feature}' not found.")

    # Filter the dataset for the selected state
    state_data = df[df['State'] == input_state]

    if state_data.empty:
        return render_template('index.html', prediction=None, error=f"No data available for the state '{input_state_name}'.")

    # Prepare the data for ARIMA model
    historical_data = state_data[input_feature]

    # Train ARIMA model on historical data for future predictions
    model_fit = train_arima_model(historical_data)

    # Predict prices for the next 7 days
    predicted_prices = predict_future_price_arima(model_fit, steps=7)

    # Display predicted prices
    prediction_dict = {f'Day {i+1}': f'{price:.2f}' for i, price in enumerate(predicted_prices)}

    # Optionally: Calculate MSE and R² score using some test data if applicable
    # For now, these metrics are not used in the ARIMA context but can be included if you have test data

    return render_template('index.html', prediction=prediction_dict)

if __name__ == "__main__":
    app.run(debug=True)
