# helpers.py

import os
import pandas as pd
import json

def load_csv(file_path):
    """ Load a CSV file into a pandas DataFrame. """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def save_csv(df, file_path):
    """ Save a pandas DataFrame to a CSV file. """
    try:
        df.to_csv(file_path, index=False)
        print(f"Successfully saved data to {file_path}")
    except Exception as e:
        print(f"Error saving file to {file_path}: {e}")

def create_directory(directory_path):
    """ Create a directory if it doesn't already exist. """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created at: {directory_path}")
    else:
        print(f"Directory already exists at: {directory_path}")

def read_json(file_path):
    """ Read data from a JSON file. """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            print(f"Successfully loaded JSON data from {file_path}")
            return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def write_json(data, file_path):
    """ Write data to a JSON file. """
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
            print(f"Successfully saved JSON data to {file_path}")
    except Exception as e:
        print(f"Error saving JSON file to {file_path}: {e}")

if __name__ == "__main__":
    # Example usage
    create_directory("example_dir")
    data = {'name': 'Rossmann', 'project': 'Sales Forecasting'}
    write_json(data, "example_dir/sample.json")
    
    # Load and save CSV example
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    save_csv(df, "example_dir/sample.csv")
    loaded_df = load_csv("example_dir/sample.csv")
    print(loaded_df)
