"""
Farmlands Analytics - Agricultural Data Science Workshop Package

This package provides comprehensive tools for agricultural supply chain data analysis,
specifically designed for Farmlands Co-operative's New Zealand operations.

Modules:
- data_cleaning: Data preprocessing and quality improvement tools
- data_exploration: Exploratory data analysis and visualization
- modeling: Machine learning models for agricultural predictions
- utils: Shared utility functions and helpers

Author: Farmlands Data Science Team
Version: 0.1.0
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Farmlands Data Science Team"
__email__ = "datascience@farmlands.co.nz"

# Import main classes for easy access
try:
    from .data_cleaning.cleaner import FarmDataCleaner
    from .data_exploration.explorer import FarmDataExplorer
    from .modeling.predictor import FarmlandsPredictiveModels
    
    __all__ = [
        'FarmDataCleaner',
        'FarmDataExplorer', 
        'FarmlandsPredictiveModels'
    ]
except ImportError:
    # Handle import errors gracefully during development
    __all__ = []