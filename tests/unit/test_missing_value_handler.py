"""
Unit tests for the agricultural missing value handler.

Tests comprehensive missing value handling functionality including:
- Automatic strategy selection
- Agricultural business logic application
- Edge case handling
- Error scenarios
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from farmlands_analytics.data_cleaning.cleaner import FarmDataCleaner


class TestMissingValueHandler(unittest.TestCase):
    """Test cases for agricultural missing value handler."""
    
    def setUp(self):
        """Set up test data for each test case."""
        self.cleaner = FarmDataCleaner("test_data")
        
        # Create test data with various missing value scenarios
        self.test_data = pd.DataFrame({
            'farm_id': ['F001', 'F002', 'F003', 'F004', 'F005', 'F006'],
            'region': ['Canterbury', 'Waikato', None, 'Otago', 'Northland', 'Bay of Plenty'],
            'product_category': ['Fertilizer', 'Seeds', 'Pesticides', None, 'Animal Feed', 'Tools'],
            'product_name': ['Urea 46%', None, 'Roundup Ready', 'Unknown Product', None, 'Spade Premium'],
            'quantity_sold': [2500.0, 150.0, None, 120.0, None, 45.0],
            'unit_price': [1.25, None, 28.90, 22.50, 12.95, None],
            'sale_date': ['2024-03-15', '2024-04-02', None, '2024-12-02', '2024-04-20', '2024-03-08'],
            'season': ['Autumn', 'Autumn', 'Autumn', 'Summer', None, 'Autumn'],
            'customer_type': ['Commercial', 'Residential', 'Commercial', 'Commercial', 'Commercial', 'Residential'],
            'weather_condition': ['Sunny', None, 'Cloudy', 'Sunny', None, None],
            'soil_type': ['Clay', 'Loam', 'Sandy', 'Clay', 'Volcanic', 'Clay']
        })
    
    def test_auto_strategy_selection(self):
        """Test automatic strategy selection based on column characteristics."""
        
        # Test the handler
        cleaned_df, report = self.cleaner.handle_missing_values(self.test_data, strategy="auto")
        
        # Verify that strategies were selected appropriately
        self.assertIn('actions_taken', report)
        
        # Check that different strategies were applied to different column types
        actions = report['actions_taken']
        
        # Product name should use agricultural product logic or unknown category
        if 'product_name' in actions:
            self.assertIn(actions['product_name']['strategy_used'], 
                         ['agricultural_product_logic', 'unknown_category'])
        
        # Numerical columns should use statistical methods
        if 'quantity_sold' in actions:
            self.assertIn(actions['quantity_sold']['strategy_used'], 
                         ['mean', 'median', 'median_with_validation'])
    
    def test_agricultural_product_logic(self):
        """Test agricultural product relationship logic."""
        
        # Create test data with missing product names but available categories
        test_data = self.test_data.copy()
        test_data.loc[test_data['product_category'] == 'Fertilizer', 'product_name'] = None
        
        cleaned_df, report = self.cleaner.handle_missing_values(test_data, strategy="auto")
        
        # Check that product names were inferred from categories
        fertilizer_rows = cleaned_df[cleaned_df['product_category'] == 'Fertilizer']
        self.assertTrue(all(fertilizer_rows['product_name'].notna()))
    
    def test_seasonal_logic(self):
        """Test New Zealand agricultural seasonal logic."""
        
        # Create test data with missing weather conditions but available seasons
        test_data = self.test_data.copy()
        test_data['weather_condition'] = None
        
        cleaned_df, report = self.cleaner.handle_missing_values(test_data, strategy="auto")
        
        # Check that weather conditions were inferred from seasons
        self.assertTrue(all(cleaned_df['weather_condition'].notna()))
        
        # Verify seasonal appropriateness
        summer_weather = cleaned_df[cleaned_df['season'] == 'Summer']['weather_condition'].iloc[0]
        self.assertIn(summer_weather, ['Sunny', 'Hot', 'Warm', 'Dry'])
    
    def test_numerical_strategy_validation(self):
        """Test numerical strategy with agricultural business validation."""
        
        test_data = self.test_data.copy()
        test_data['quantity_sold'] = [10000, 5, None, None, None, 2]  # Mixed extreme values
        
        cleaned_df, report = self.cleaner.handle_missing_values(test_data, strategy="auto")
        
        # Verify all missing values were filled
        self.assertTrue(all(cleaned_df['quantity_sold'].notna()))
        
        # Check that validation was applied
        if 'quantity_sold' in report['actions_taken']:
            action = report['actions_taken']['quantity_sold']
            self.assertEqual(action['missing_before'], 3)
            self.assertEqual(action['missing_after'], 0)
    
    def test_different_strategies(self):
        """Test different manual strategies."""
        
        strategies_to_test = ['mean', 'median', 'mode', 'forward_fill']
        
        for strategy in strategies_to_test:
            with self.subTest(strategy=strategy):
                cleaned_df, report = self.cleaner.handle_missing_values(self.test_data, strategy=strategy)
                
                # Check that data was processed
                self.assertIsNotNone(cleaned_df)
                self.assertIn('actions_taken', report)
                
                # Verify completeness improved
                original_completeness = (self.test_data.count().sum() / 
                                       (self.test_data.shape[0] * self.test_data.shape[1])) * 100
                final_completeness = (cleaned_df.count().sum() / 
                                    (cleaned_df.shape[0] * cleaned_df.shape[1])) * 100
                
                self.assertGreaterEqual(final_completeness, original_completeness)
    
    def test_edge_cases(self):
        """Test edge cases and error scenarios."""
        
        # Test with all missing values in a column
        test_data = self.test_data.copy()
        test_data['new_column'] = None
        
        cleaned_df, report = self.cleaner.handle_missing_values(test_data, strategy="auto")
        
        # Should handle gracefully
        self.assertIsNotNone(cleaned_df)
        
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        cleaned_empty, empty_report = self.cleaner.handle_missing_values(empty_df, strategy="auto")
        
        self.assertTrue(cleaned_empty.empty)
        self.assertIn('actions_taken', empty_report)
    
    def test_agricultural_logic_validation(self):
        """Test agricultural business logic validation."""
        
        cleaned_df, report = self.cleaner.handle_missing_values(self.test_data, strategy="auto")
        
        # Check that agricultural logic was applied
        self.assertIn('agricultural_logic_applied', report)
        self.assertIsInstance(report['agricultural_logic_applied'], list)
        
        # Verify data types were validated
        if 'sale_date' in cleaned_df.columns:
            # Should convert to datetime where possible
            for date_val in cleaned_df['sale_date'].dropna():
                try:
                    pd.to_datetime(date_val)
                    date_valid = True
                except:
                    date_valid = False
                # Allow both datetime and string representations
                self.assertTrue(date_valid or isinstance(date_val, str))
    
    def test_completeness_improvement(self):
        """Test that completeness actually improves."""
        
        cleaned_df, report = self.cleaner.handle_missing_values(self.test_data, strategy="auto")
        
        # Calculate original completeness
        original_missing = self.test_data.isnull().sum().sum()
        original_total = self.test_data.shape[0] * self.test_data.shape[1]
        original_completeness = ((original_total - original_missing) / original_total) * 100
        
        # Calculate final completeness
        final_missing = cleaned_df.isnull().sum().sum()
        final_total = cleaned_df.shape[0] * cleaned_df.shape[1]
        final_completeness = ((final_total - final_missing) / final_total) * 100
        
        # Completeness should improve or stay the same
        self.assertGreaterEqual(final_completeness, original_completeness)
        
        # Check report matches calculation
        self.assertAlmostEqual(
            report['completeness_improvement'], 
            final_completeness - original_completeness,
            places=1
        )
    
    def test_critical_columns_handling(self):
        """Test handling of critical agricultural business columns."""
        
        # Test with missing critical columns
        test_data = self.test_data.copy()
        test_data.loc[0, 'farm_id'] = None  # Critical business identifier
        
        cleaned_df, report = self.cleaner.handle_missing_values(test_data, strategy="auto")
        
        # Critical columns should be handled appropriately
        # Either filled or flagged in warnings
        if cleaned_df['farm_id'].isnull().any():
            self.assertIn('warnings', report)
        else:
            # Should be filled with appropriate value
            self.assertTrue(all(cleaned_df['farm_id'].notna()))
    
    def test_report_structure(self):
        """Test that the report contains all expected elements."""
        
        cleaned_df, report = self.cleaner.handle_missing_values(self.test_data, strategy="auto")
        
        # Check report structure
        expected_keys = [
            'original_shape', 'columns_processed', 'actions_taken',
            'missing_patterns', 'agricultural_logic_applied', 
            'warnings', 'final_shape', 'completeness_improvement'
        ]
        
        for key in expected_keys:
            self.assertIn(key, report, f"Missing key in report: {key}")
        
        # Check action details structure
        for column, action in report['actions_taken'].items():
            self.assertIn('strategy_used', action)
            self.assertIn('missing_before', action)
            self.assertIn('action_details', action)


class TestMissingPatternAnalysis(unittest.TestCase):
    """Test missing pattern analysis functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.cleaner = FarmDataCleaner("test_data")
    
    def test_completeness_categorization(self):
        """Test completeness level categorization."""
        
        # Test different completeness levels
        test_cases = [
            (0, "complete"),
            (3, "excellent"),
            (10, "good"),
            (25, "moderate"),
            (60, "poor")
        ]
        
        for missing_pct, expected_category in test_cases:
            category = self.cleaner._categorize_completeness(missing_pct)
            self.assertEqual(category, expected_category)
    
    def test_pattern_analysis(self):
        """Test missing pattern analysis."""
        
        # Create test data with known patterns
        test_data = pd.DataFrame({
            'complete_col': [1, 2, 3, 4, 5],
            'mostly_complete': [1, 2, None, 4, 5],  # 20% missing
            'half_missing': [1, None, None, 4, None],  # 60% missing
            'categorical': ['A', 'B', None, 'A', 'B'],
            'numerical': [1.1, 2.2, None, 4.4, 5.5]
        })
        
        # Use the private method for testing
        import logging
        logger = logging.getLogger(__name__)
        patterns = self.cleaner._analyze_missing_patterns(test_data, logger)
        
        # Check pattern analysis results
        self.assertEqual(patterns['complete_col']['completeness_category'], 'complete')
        self.assertEqual(patterns['mostly_complete']['completeness_category'], 'moderate')  # 20% missing = moderate
        self.assertEqual(patterns['half_missing']['completeness_category'], 'poor')


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
