import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from src.model_training import train_model
import joblib
import os

class TestModelTraining(unittest.TestCase):
    
    def setUp(self):
        """Set up a small dataset for testing model training."""
        data = {
            'Store': [1, 2, 3, 4, 5],
            'Sales': [500, 600, 550, 450, 700],
            'Customers': [50, 60, 55, 45, 70],
            'DayOfWeek': [1, 2, 3, 4, 5],
            'CompetitionDistance': [100, 200, 150, 300, 400]
        }
        self.df = pd.DataFrame(data)
        self.features = ['Customers', 'DayOfWeek', 'CompetitionDistance']
        self.target = 'Sales'
    
    def test_train_model(self):
        """Test if the train_model function trains and saves a Random Forest model correctly."""
        X = self.df[self.features]
        y = self.df[self.target]
        
        # Call the train_model function
        model_filename = train_model(X, y)
        
        # Check if the model file was saved
        self.assertTrue(os.path.exists(model_filename), "Model was not saved correctly.")
        
        # Load the model and check if it's a RandomForestRegressor
        model = joblib.load(model_filename)
        self.assertIsInstance(model, RandomForestRegressor, "The model is not a RandomForestRegressor.")
        
        # Clean up (remove the model file)
        os.remove(model_filename)

if __name__ == '__main__':
    unittest.main()
