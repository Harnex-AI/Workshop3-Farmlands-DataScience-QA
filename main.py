"""
Farmlands Analytics Workshop - Main Entry Point

This script demonstrates the basic usage of all workshop modules and provides
a comprehensive overview of the workshop structure and available functionality.
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from farmlands_analytics.data_cleaning.cleaner import FarmDataCleaner
    from farmlands_analytics.data_exploration.explorer import FarmDataExplorer
    from farmlands_analytics.modeling.predictor import FarmlandsPredictiveModels
    print("✅ Successfully imported all Farmlands Analytics modules")
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("This is expected if you haven't implemented the TODO sections yet!")


def main() -> None:
    """
    Main function that displays workshop overview and system status.
    
    This function provides a comprehensive overview of the workshop structure,
    checks for data availability, validates module installation, and provides
    quick start instructions for participants.
    """
    print("🌾 Welcome to Farmlands Analytics Workshop!")
    print("=" * 50)
    
    # Check data availability
    _check_data_availability()
    
    # Check workshop modules
    _check_workshop_modules()
    
    # Check test suites
    _check_test_suites()
    
    # Display quick start guide
    _display_quick_start_guide()
    
    # Display AI prompt examples
    _display_ai_prompt_examples()
    
    print("\n📚 For detailed instructions, see README.md")
    print("\n🤖 Happy AI-assisted coding!")


def _check_data_availability() -> None:
    """
    Check and display available sample datasets.
    
    Verifies that the required CSV data files are present in the data/raw
    directory and displays their status to the user.
    """
    print("\n📊 Sample Data Available:")
    data_dir = Path("data/raw")
    
    if not data_dir.exists():
        print("  ⚠️ Data directory not found")
        return
    
    csv_files = list(data_dir.glob("*.csv"))
    if csv_files:
        for file in csv_files:
            print(f"  ✓ {file.name}")
    else:
        print("  ⚠️ No CSV files found in data directory")


def _check_workshop_modules() -> None:
    """
    Check and display status of workshop modules.
    
    Verifies that all core workshop modules are present and displays
    their file paths and availability status.
    """
    print("\n🎯 Workshop Modules:")
    modules = [
        ("Data Cleaning", "src/farmlands_analytics/data_cleaning/cleaner.py"),
        ("Data Exploration", "src/farmlands_analytics/data_exploration/explorer.py"),
        ("ML Modeling", "src/farmlands_analytics/modeling/predictor.py"),
    ]
    
    for name, path in modules:
        if Path(path).exists():
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: {path} (not found)")


def _check_test_suites() -> None:
    """
    Check and display status of test suites.
    
    Verifies that all test files are present for unit and integration
    testing of the workshop components.
    """
    print("\n🧪 Test Suites:")
    test_files = [
        "tests/unit/test_data_cleaning.py",
        "tests/unit/test_data_exploration.py", 
        "tests/integration/test_end_to_end_pipeline.py"
    ]
    
    for test_file in test_files:
        if Path(test_file).exists():
            print(f"  ✓ {test_file}")
        else:
            print(f"  ✗ {test_file} (not found)")


def _display_quick_start_guide() -> None:
    """
    Display quick start instructions for workshop participants.
    
    Provides step-by-step guidance for getting started with the workshop
    activities and using Cursor AI for implementation.
    """
    print("\n🚀 Quick Start Guide:")
    print("  1. Open Cursor IDE")
    print("  2. Navigate to src/farmlands_analytics/data_cleaning/cleaner.py")
    print("  3. Find a TODO section")
    print("  4. Use Cursor AI to implement the function")
    print("  5. Test your implementation: pytest tests/unit/ -v")


def _display_ai_prompt_examples() -> None:
    """
    Display example AI prompts for workshop participants.
    
    Provides concrete examples of effective AI prompts that participants
    can use to implement workshop functions using Cursor AI.
    """
    print("\n💡 AI Prompt Example:")
    print("  'Implement a missing value handler for agricultural data that chooses")
    print("   the best strategy based on column type and missing percentage'")
    print("\n💡 Additional Prompt Ideas:")
    print("  'Create seasonal analysis for New Zealand agricultural products'")
    print("  'Build demand forecasting model with weather correlation'")
    print("  'Generate customer segmentation using RFM analysis'")


if __name__ == "__main__":
    main()
