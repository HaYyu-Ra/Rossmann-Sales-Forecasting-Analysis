# test_model_baseline.py

import unittest
import os
from models.model_baseline import train_baseline_model

class TestModelBaseline(unittest.TestCase):

    def setUp(self):
        # Paths for test data and output
        self.data_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\train.csv'
        self.model_output_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\models\saved_models\test_baseline_model.pkl'
    
    def test_train_baseline_model(self):
        # Train a baseline model and check if the model file is created
        train_baseline_model(self.data_path, self.model_output_path)
        self.assertTrue(os.path.exists(self.model_output_path))

    def tearDown(self):
        # Remove the test model file after testing
        if os.path.exists(self.model_output_path):
            os.remove(self.model_output_path)

if __name__ == "__main__":
    unittest.main()
