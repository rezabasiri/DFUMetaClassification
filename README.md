# Diabetic Foot Ulcer (DFU) Classification

This project implements a machine learning pipeline for classifying Diabetic Foot Ulcers (DFUs) using various classification algorithms and feature selection techniques.

## Project Description

The pipeline includes data preprocessing, feature engineering, SHAP-based feature selection, and a leave-one-patient-out cross-validation approach. It implements multiple classification algorithms including Ordinal Random Forest, SVM, XGBoost, Neural Networks, and CatBoost.

## Requirements

- Python 3.9.19
- scikit-learn 1.5
- XGBoost 2.0.3
- CatBoost 1.2.3
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/dfu-classification.git
   cd dfu-classification
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the main script:

```
python main.py
```

## Project Structure

- `main.py`: The main script that orchestrates the entire process
- `data_preprocessing.py`: Functions for data loading and preprocessing
- `feature_engineering.py`: Functions for feature engineering and selection
- `models.py`: Defines the custom models and parameter grids
- `utils.py`: Contains utility functions used across the project
- `cross_validation.py`: Implements the leave-one-patient-out cross-validation

## License

MIT

## Contributing

Reza Basiri, PhD University of Toronto
PIs: Dr. Milos R. Popovic, Dr. Shehroz S. Khan