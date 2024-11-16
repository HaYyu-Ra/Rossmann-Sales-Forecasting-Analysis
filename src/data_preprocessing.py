import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def preprocess_data(train_data):
    logging.info("Starting data preprocessing...")

    # Check available columns
    logging.info(f"Available columns: {train_data.columns.tolist()}")

    # Fill missing values for existing columns
    if 'SchoolHoliday' in train_data.columns:
        train_data['SchoolHoliday'] = train_data['SchoolHoliday'].fillna(0).astype(int)

    # Feature engineering
    train_data['Date'] = pd.to_datetime(train_data['Date'])
    train_data['Weekday'] = train_data['Date'].dt.weekday
    train_data['IsWeekend'] = (train_data['Weekday'] >= 5).astype(int)
    train_data['Day'] = train_data['Date'].dt.day
    train_data['Month'] = train_data['Date'].dt.month
    train_data['Year'] = train_data['Date'].dt.year
    train_data['DaysToHoliday'] = (pd.to_datetime('2023-12-25') - train_data['Date']).dt.days

    # One-hot encoding for categorical columns
    train_data = pd.get_dummies(train_data, columns=['StateHoliday', 'Promo'], drop_first=True)

    # Convert 'Date' to numeric (timestamp)
    train_data['Date'] = train_data['Date'].astype(np.int64) // 10**9  # Convert to seconds since epoch

    logging.info("Data preprocessing complete.")
    return train_data

if __name__ == "__main__":
    # Load the dataset
    train_data = pd.read_csv(r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\train.csv", low_memory=False)

    # Preprocess the data
    processed_data = preprocess_data(train_data)

    # Optionally, save the processed data to a new CSV
    processed_data.to_csv(r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\processed_train.csv", index=False)

    logging.info("Processed data saved successfully.")
