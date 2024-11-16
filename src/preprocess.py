# preprocess.py

import pandas as pd

# Input and Output paths for data files
input_paths = {
    'clean_data.csv': r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\clean_data.csv',
    'sample_submission.csv': r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\sample_submission.csv',
    'store.csv': r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\store.csv',
    'synthetic_sales_data.csv': r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\synthetic_sales_data.csv',
    'test.csv': r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\test.csv',
    'train.csv': r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\train.csv',
}

output_paths = {
    'clean_data.csv': r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\clean_data_cleaned.csv',
    'sample_submission.csv': r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\sample_submission_cleaned.csv',
    'store.csv': r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\store_cleaned.csv',
    'synthetic_sales_data.csv': r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\synthetic_sales_data_cleaned.csv',
    'test.csv': r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\test_cleaned.csv',
    'train.csv': r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\train_cleaned.csv',
}

def load_data(file_path):
    """ Load data from a CSV file. """
    return pd.read_csv(file_path)

def clean_data(df):
    """ Perform data cleaning operations. """
    # Example of cleaning: dropping nulls and resetting index
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def save_data(df, file_path):
    """ Save cleaned data to a CSV file. """
    df.to_csv(file_path, index=False)

def preprocess_data():
    """ Load, clean, and save all data files. """
    for file_name, input_path in input_paths.items():
        print(f"Processing {file_name}...")
        df = load_data(input_path)
        cleaned_df = clean_data(df)
        output_path = output_paths[file_name]
        save_data(cleaned_df, output_path)
        print(f"Saved cleaned data to {output_path}")

if __name__ == "__main__":
    preprocess_data()
