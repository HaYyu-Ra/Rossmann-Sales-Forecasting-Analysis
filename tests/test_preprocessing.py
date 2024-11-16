# test_preprocessing.py

import unittest
import pandas as pd
from data_processing.preprocess import clean_data

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Sample data setup for testing
        self.sample_data = pd.DataFrame({
            'Store': [1, 2],
            'DayOfWeek': [1, 4],
            'Sales': [5000, 7000],
            'Customers': [550, 600],
            'Open': [1, 1],
            'Promo': [1, 0],
            'StateHoliday': ['0', '0'],
            'SchoolHoliday': [0, 1]
        })
    
    def test_clean_data(self):
        # Test the clean_data function
        cleaned_data = clean_data(self.sample_data)
        # Check if null values are handled
        self.assertFalse(cleaned_data.isnull().values.any())
        # Check if the columns are correct after cleaning
        expected_columns = ['Store', 'DayOfWeek', 'Sales', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']
        self.assertListEqual(list(cleaned_data.columns), expected_columns)

if __name__ == "__main__":
    unittest.main()
