#!/usr/bin/env python3
"""
Main pipeline script for income classification and segmentation project.
This script runs the complete analysis pipeline from data processing to model evaluation.
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def run_script(script_path, description):
    """Run a Python script and handle any errors."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            [sys.executable, script_path], capture_output=True, text=True, check=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")

    # Map package names to their import names
    package_mapping = {
        "pandas": "pandas",
        "numpy": "numpy",
        "scikit-learn": "sklearn",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "xgboost": "xgboost",
        "imbalanced-learn": "imblearn",
    }

    missing_packages = []
    for package, import_name in package_mapping.items():
        try:
            __import__(import_name)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            missing_packages.append(package)

    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    print("\nAll required packages are installed.")
    return True


def check_data_files():
    """Check if real data files exist."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Check if real data files exist
    data_file = data_dir / "census-bureau.data"
    columns_file = data_dir / "census-bureau.columns"

    if data_file.exists() and columns_file.exists():
        print(f"Real data files found:")
        print(
            f"   Data file: {data_file} ({data_file.stat().st_size / 1024/1024:.1f} MB)"
        )
        print(f"   Columns file: {columns_file}")
        return True
    else:
        print("Real data files not found.")
        print("Please ensure the following files exist in the data/ directory:")
        print("   - census-bureau.data")
        print("   - census-bureau.columns")
        return False


def main():
    """Main function to run the complete pipeline."""
    print("INCOME CLASSIFICATION AND SEGMENTATION PIPELINE")
    print("=" * 60)
    print("This script will run the complete analysis pipeline:")
    print("1. Data processing and exploration")
    print("2. Income classification model training")
    print("3. Customer segmentation model training")
    print("4. Generate results and visualizations")
    print("=" * 60)

    # Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies and try again.")
        return

    # Check for real data files
    if not check_data_files():
        print("Please add the required data files and try again.")
        return

    # Define script paths
    scripts = [
        ("src/data_processing.py", "Data Processing and Exploration"),
        ("src/classification_model.py", "Income Classification Model Training"),
        ("src/segmentation_model.py", "Customer Segmentation Model Training"),
    ]

    # Run each script
    start_time = time.time()
    success_count = 0

    for script_path, description in scripts:
        if run_script(script_path, description):
            success_count += 1
            print(f"✓ {description} completed successfully")
        else:
            print(f"✗ {description} failed")
            break

    # Calculate total time
    total_time = time.time() - start_time

    # Summary
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Scripts completed: {success_count}/{len(scripts)}")

    if success_count == len(scripts):
        print("\nPIPELINE COMPLETED SUCCESSFULLY!")
        print("\nGenerated outputs:")
        print("- Trained models saved in 'models/' directory")
        print("- Results and visualizations saved in 'results/' directory")
        print("- Project report available as 'project_report.md'")
        print("\nNext steps:")
        print("1. Review the project report for detailed analysis")
        print("2. Check the 'results/' directory for visualizations")
        print("3. Use the trained models for predictions")
        print("4. Customize marketing strategies based on segment analysis")
    else:
        print(f"\nPIPELINE INCOMPLETE: {len(scripts) - success_count} scripts failed")
        print("Please check the error messages above and fix any issues.")


if __name__ == "__main__":
    main()
