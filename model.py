import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
import xgboost as xgb
import pickle
from statsmodels.tsa.arima.model import ARIMA

# Load and preprocess the dataset
df = pd.read_excel('/content/combined.xlsx')

# Fill missing values
columns_to_fill = ['Wheat', 'Gramdal', 'Turdal', 'Uraddal', 'Milk', 'Groundnutoil', 'Soyaoil', 'Sunfloweroil', 'Palmoil', 'Saltpack', 'Vanaspati', 'Rice', 'Atta', 'Moongdal', 'Masoordal', 'Mustardoil', 'Sugar', 'Tealoose', 'Gur', 'Potato', 'Onion', 'Tomato']
for column in columns_to_fill:
    median_value = df[column].median()
    df[column].fillna(median_value, inplace=True)

# Encode categorical data
label_encode = LabelEncoder()
df['State'] = label_encode.fit_transform(df['State'])

# List of relevant features
features = [
    'Rice', 'Wheat', 'Atta', 'Gramdal', 'Turdal', 'Uraddal', 'Moongdal', 'Masoordal',
    'Sugar', 'Milk', 'Groundnutoil', 'Mustardoil', 'Vanaspati', 'Soyaoil', 'Sunfloweroil',
    'Palmoil', 'Gur', 'Tealoose', 'Saltpack', 'Potato', 'Onion', 'Tomato'
]

def train_arima_model(data, order=(5, 1, 0)):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit

def predict_future_price_arima(model_fit, steps=7):
    forecast = model_fit.forecast(steps=steps)
    return forecast

if __name__ == "__main__":
    input_state_name = input("Enter the state name: ")
    input_feature = input("Enter the feature name: ")

    if input_state_name not in label_encode.classes_:
        print(f"State '{input_state_name}' not found.")
        exit()

    input_state = label_encode.transform([input_state_name])[0]

    if input_feature not in features:
        print(f"Feature '{input_feature}' not found.")
        exit()

    state_data = df[df['State'] == input_state]

    if state_data.empty:
        print(f"No data available for the state '{input_state_name}'.")
        exit()

    # Drop DateTime columns if they exist
    if 'Date' in state_data.columns:
        state_data = state_data.drop(columns=['Date'])

    # Prepare the training data (drop the target column)
    X = state_data.drop(columns=[input_feature])
    y = state_data[input_feature]

    # Ensure no DateTime types in feature set
    X = X.select_dtypes(include=[np.number])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Base Models
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gbr', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42)),
    ]

    # Meta-Model
    meta_model = Ridge(alpha=1.0)

    # Stacking Regressor
    stacking_regressor = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5
    )

    # Train Stacking Regressor
    stacking_regressor.fit(X_train, y_train)

    # Predict on the test set
    y_pred = stacking_regressor.predict(X_test)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error of Stacking Model: {mse:.2f}')

    # Calculate R² score
    r2 = r2_score(y_test, y_pred)
    print(f'R² Score of Stacking Model: {r2:.4f}')

    # Train ARIMA model on historical data for future predictions
    model_fit = train_arima_model(state_data[input_feature])

    # Predict prices for the next 7 days
    predicted_prices = predict_future_price_arima(model_fit, steps=7)

    # Display predicted prices
    print(f'Predicted Prices for the next 7 days for {input_feature} (Unit: ₹/Kg):')
    for day, price in enumerate(predicted_prices, start=1):
        print(f'Day {day}: ₹{price:.2f}/Kg')

    # Save the trained stacking regressor model
    pickle.dump(stacking_regressor, open("model.pkl", "wb"))
