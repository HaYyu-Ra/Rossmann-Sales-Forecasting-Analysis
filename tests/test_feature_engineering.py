# test_feature_engineering.py

import unittest
import pandas as pd
from data_processing.feature_engineering import add_features

class TestFeatureEngineering(unittest.TestCase):

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
            'SchoolHoliday': [0, 1],
            'Date': ['2023-11-01', '2023-11-04']
        })
        self.sample_data['Date'] = pd.to_datetime(self.sample_data['Date'])
    
    def test_add_features(self):
        # Test the add_features function
        feature_data = add_features(self.sample_data)
        # Check if the feature engineering added the new columns
        self.assertIn('Year', feature_data.columns)
        self.assertIn('Month', feature_data.columns)
        self.assertIn('Day', feature_data.columns)
        self.assertIn('WeekOfYear', feature_data.columns)

if __name__ == "__main__":
    unittest.main()
