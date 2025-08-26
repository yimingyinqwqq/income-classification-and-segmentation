# Income Classification and Segmentation Project

## Project Overview
This project implements two machine learning models for a retail business client:
1. **Income Classification Model**: Predicts whether individuals earn above or below $50,000
2. **Customer Segmentation Model**: Creates marketing segments based on demographic and employment characteristics

### Important Note
While I have experience with data analysis and basic machine learning concepts, I acknowledge that lots of code and techniques were implemented with assistance from AI tools.

## Project Structure
```
income-classification-and-segmentation/
├── data/                         # Data files
│   ├── censusbureau.data         # Main dataset (40 variables + weight + label)
│   └── census-bureau.columns     # Column headers
├── src/                          # Source code
│   ├── data_processing.py        # Data loading and preprocessing
│   ├── classification_model.py   # Income classification implementation
│   ├── segmentation_model.py     # Customer segmentation implementation
│   └── utils.py                  # Utility functions
├── models/                       # Trained models
├── results/                      # Output files and visualizations
│   ├── classification/          # Income classification results
│   └── segmentation/            # Customer segmentation results
├── requirements.txt              # Python dependencies
└── report.pdf                    # Client report
```

## Setup Instructions

### Prerequisites
- Python 3.12 (recommended) or Python 3.8+
- pip (Python package installer)

### Installation

#### Step 1: Setup Project
```bash
cd income-classification-and-segmentation

# Create virtual environment
python3.12 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Install required packages
pip install -r requirements.txt
```

### Running the Project

#### Option 1: Run via Python Scripts
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Data processing and exploration
python src/data_processing.py

# Train and evaluate classification model
python src/classification_model.py

# Generate segmentation model
python src/segmentation_model.py
```


## Expected Outputs

### Classification Results (`results/classification/`)
- Model comparison charts and metrics
- Feature importance plots
- Income distribution analysis
- Correlation matrix

### Segmentation Results (`results/segmentation/`)
- Segment distribution charts
- Income distribution by segment
- Age distribution by segment
- Optimal cluster analysis
- Marketing recommendations (JSON)
- Segment analysis data (CSV)

### Models
- Trained classification model saved in `models/`
- Segmentation model saved in `models/`

## Data Requirements
The project expects census data with 40 demographic and employment variables, plus:
- Weight column: Relative distribution weight for each observation
- Label column: Income classification (>$50k or ≤$50k)

## Package Versions
This project uses the latest stable versions of key packages:
- Python 3.12.11
- NumPy 2.3.2
- Pandas 2.3.2
- Scikit-learn 1.7.1
- XGBoost 3.0.4
- Matplotlib 3.10.5
- Seaborn 0.13.2

## Model Performance
- Classification Model: Evaluated using accuracy, precision, recall, F1-score, and ROC-AUC
- Segmentation Model: Evaluated using silhouette score and business interpretability
