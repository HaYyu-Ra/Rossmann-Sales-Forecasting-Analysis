# feature_engineering.py

import pandas as pd

def add_date_features(df, date_column='Date'):
    """ Add date-based features such as year, month, day, and day of the week. """
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df['Year'] = df[date_column].dt.year
        df['Month'] = df[date_column].dt.month
        df['Day'] = df[date_column].dt.day
        df['DayOfWeek'] = df[date_column].dt.dayofweek
    return df

def add_sales_features(df):
    """ Add sales-related features like rolling average and cumulative sum. """
    if 'Sales' in df.columns:
        df['SalesRollingMean'] = df['Sales'].rolling(window=7).mean()  # 7-day rolling average
        df['CumulativeSales'] = df['Sales'].cumsum()  # Cumulative sales
    return df

def perform_feature_engineering(input_path, output_path):
    """ Load data, add features, and save enhanced data. """
    df = pd.read_csv(input_path)
    df = add_date_features(df)
    df = add_sales_features(df)
    df.to_csv(output_path, index=False)
    print(f"Feature engineered data saved to {output_path}")

if __name__ == "__main__":
    # Example of feature engineering for 'train.csv'
    input_file = input_paths['train.csv']
    output_file = output_paths['train.csv']
    perform_feature_engineering(input_file, output_file)
