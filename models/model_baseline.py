# model_baseline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

def train_baseline_model(data_path, output_model_path):
    """ Train a baseline model using a RandomForestRegressor. """
    
    # Load the dataset
    data = pd.read_csv(data_path)
    
    # Assuming 'Sales' is the target variable
    X = data.drop(columns=['Sales'])
    y = data['Sales']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predictions
    predictions = model.predict(X_test)
    
    # Calculate RMSE
    rmse = mean_squared_error(y_test, predictions, squared=False)
    print(f"Baseline Model RMSE: {rmse}")
    
    # Save the trained model
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump(model, output_model_path)
    print(f"Model saved to {output_model_path}")

if __name__ == "__main__":
    # Example usage
    data_file_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\train.csv'
    model_output_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\models\saved_models\baseline_model.pkl'
    train_baseline_model(data_file_path, model_output_path)
