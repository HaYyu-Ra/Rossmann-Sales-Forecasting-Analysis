# model_lstm.py

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def create_lstm_model(input_shape):
    """ Create an LSTM model for sales forecasting. """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Output layer for regression
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_data(data_path):
    """ Load and prepare data for LSTM training. """
    
    # Load the dataset
    data = pd.read_csv(data_path)
    
    # Assume 'Sales' is the target and 'Date' is a time index column
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Sales']])
    
    # Save the scaler for future use
    scaler_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\models\saved_models\minmax_scaler.pkl'
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    
    # Prepare the dataset for supervised learning (sliding window)
    X = []
    y = []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])  # 60 time steps look back
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM input
    
    return X, y, scaler

def train_lstm_model(data_path, model_output_path):
    """ Train an LSTM model for sales forecasting. """
    
    # Prepare the data
    X, y, scaler = prepare_data(data_path)
    
    # Create LSTM model
    model = create_lstm_model(input_shape=(X.shape[1], 1))
    
    # Train the model
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)
    
    # Save the trained model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    model.save(model_output_path)
    print(f"LSTM Model saved to {model_output_path}")

if __name__ == "__main__":
    # Example usage
    data_file_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\train.csv'
    lstm_model_output_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\models\saved_models\lstm_model.h5'
    train_lstm_model(data_file_path, lstm_model_output_path)
