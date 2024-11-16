# test_model_lstm.py

import unittest
import os
from models.model_lstm import train_lstm_model

class TestModelLSTM(unittest.TestCase):

    def setUp(self):
        # Paths for test data and output
        self.data_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\train.csv'
        self.model_output_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\models\saved_models\test_lstm_model.h5'
    
    def test_train_lstm_model(self):
        # Train an LSTM model and check if the model file is created
        train_lstm_model(self.data_path, self.model_output_path)
        self.assertTrue(os.path.exists(self.model_output_path))

    def tearDown(self):
        # Remove the test model file after testing
        if os.path.exists(self.model_output_path):
            os.remove(self.model_output_path)

if __name__ == "__main__":
    unittest.main()
