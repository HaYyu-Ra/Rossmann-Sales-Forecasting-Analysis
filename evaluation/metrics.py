# metrics.py

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_mae(y_true, y_pred):
    """ Calculate Mean Absolute Error (MAE). """
    return mean_absolute_error(y_true, y_pred)

def calculate_mse(y_true, y_pred):
    """ Calculate Mean Squared Error (MSE). """
    return mean_squared_error(y_true, y_pred)

def calculate_rmse(y_true, y_pred):
    """ Calculate Root Mean Squared Error (RMSE). """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_r2_score(y_true, y_pred):
    """ Calculate R-squared (coefficient of determination). """
    return r2_score(y_true, y_pred)

def evaluate_model(y_true, y_pred):
    """ Evaluate model performance with various metrics. """
    metrics = {
        'MAE': calculate_mae(y_true, y_pred),
        'MSE': calculate_mse(y_true, y_pred),
        'RMSE': calculate_rmse(y_true, y_pred),
        'R^2 Score': calculate_r2_score(y_true, y_pred)
    }
    return metrics

if __name__ == "__main__":
    # Example usage
    y_true = [100, 200, 300, 400, 500]  # Example true values
    y_pred = [110, 190, 310, 405, 495]  # Example predicted values
    
    metrics = evaluate_model(y_true, y_pred)
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
