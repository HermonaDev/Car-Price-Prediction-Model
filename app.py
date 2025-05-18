from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
import requests
from datetime import datetime
import os

app = Flask(__name__)

# Function to get current USD to ETB exchange rate
def get_usd_to_etb_rate():
    try:
        # Get API key from environment variable or use the hardcoded one as fallback
        api_key = os.environ.get('EXCHANGE_RATE_API_KEY', '1a0cec85dba024b67760288b')
        response = requests.get(f'https://v6.exchangerate-api.com/v6/{api_key}/latest/USD')
        if response.status_code == 200:
            data = response.json()
            return data['conversion_rates']['ETB']
        return None
    except:
        return None

# Convert lakh INR to USD (approximate conversion: 1 lakh INR ≈ 1200 USD as of 2024)
def lakh_inr_to_usd(lakh_value):
    inr_value = lakh_value * 100000  # Convert lakh to INR
    usd_value = inr_value / 83  # Approximate INR to USD rate as of 2024
    return usd_value

def train_model():
    df = pd.read_csv('car-dataset.csv')
    df = df.drop(['Car_Name', 'Seller_Type', 'Present_Price'], axis=1)
    
    # Convert Selling_Price from lakh INR to USD
    df['Selling_Price'] = df['Selling_Price'].apply(lakh_inr_to_usd)
    
    # Prepare features
    X = df[['Transmission', 'Fuel_Type', 'Kms_Driven', 'Year', 'Owner']]
    y = df['Selling_Price']
    
    # Convert categorical variables
    X = pd.get_dummies(X, drop_first=True)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save the model
    joblib.dump(model, 'models/car_price_model.pkl')
    # Save the feature names
    joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')
    
    return model, X.columns.tolist()

# Train and save the model on startup
model, feature_names = train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        transmission = request.form['transmission']
        fuel_type = request.form['fuel_type']
        kms_driven = float(request.form['kms_driven'])
        year = int(request.form['year'])
        owner = request.form['owner']
        
        # Create a DataFrame with the input
        input_data = pd.DataFrame({
            'Transmission': [transmission],
            'Fuel_Type': [fuel_type],
            'Kms_Driven': [kms_driven],
            'Year': [year],
            'Owner': [owner]
        })
        
        # Convert to dummy variables
        input_encoded = pd.get_dummies(input_data)
        
        # Align input features with training features
        final_input = pd.DataFrame(columns=feature_names)
        for column in input_encoded.columns:
            if column in feature_names:
                final_input[column] = input_encoded[column]
            
        # Fill missing columns with 0
        final_input = final_input.fillna(0)
        
        # Make prediction (this will be in USD)
        prediction_usd = model.predict(final_input)[0]
        
        # Get current USD to ETB rate
        current_rate = get_usd_to_etb_rate()
        
        if current_rate:
            prediction_etb = prediction_usd * current_rate
            etb_message = f"ETB {prediction_etb:,.2f} (at current rate: 1 USD = {current_rate:.2f} ETB)"
        else:
            # Fallback to approximate rate if API fails
            prediction_etb = prediction_usd * 55.85  # Example fallback rate
            etb_message = f"ETB {prediction_etb:,.2f} (approximate)"
        
        return jsonify({
            'success': True,
            'prediction_dollar': f"${prediction_usd:,.2f}",
            'prediction_birr': etb_message,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True) 