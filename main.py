import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from data_preprocessing import load_and_preprocess_data
from feature_engineering import engineer_features, calculate_shap_values, select_top_features
from models import get_classifiers, get_param_grids
from utils import apply_sampling, transform_labels
from cross_validation import process_patient

# Constants
DATA_FILE = 'CSVFile.csv' # Path to the dataset
TARGET_CLASS = 'Healing Phase' # Target class to predict
N_FEATURES = 22 # Number of features to select, can be a list of values to test different feature selection methods

def main():
    # Load and preprocess data
    data = load_and_preprocess_data(DATA_FILE, TARGET_CLASS)
    
    # Engineer features
    data = engineer_features(data)
    
    # Define patient numbers and selected methods
    patient_numbers = data['Patient#'].unique()
    selected_methods = [
        ['Ordinal Random Forest'], 
                        ['SVM'], 
                        ['XGBoost'], 
                        ['Neural Network'], 
                        ['CatBoost']
                        ] # List of classifiers to test, can be a list of dictionaries to test different clf1-clf2 combinations
    
    # Initialize results dictionary
    local_results_all = initialize_results(selected_methods)
    
    # Perform leave-one-patient-out cross-validation
    patient_numbers_pbar = tqdm(patient_numbers, desc="Processing patients")
    results = Parallel(n_jobs=-1)(delayed(process_patient)(
        patient, data, TARGET_CLASS, N_FEATURES, selected_methods
    ) for patient in patient_numbers_pbar)
    
    # Process and analyze results
    process_results(results, local_results_all)
    
    print("Cross-validation completed.")

def initialize_results(selected_methods):
    local_results_all = {}
    for clf_config in selected_methods:
        if isinstance(clf_config, dict):
            clf_names = [f"{config_name}_{'_'.join(clf_config[config_name])}" for config_name in clf_config]
            clf_name = '_'.join(clf_names)
        else:
            clf_name = '_'.join(clf_config) if isinstance(clf_config, list) else clf_config
        
        local_results_all[clf_name] = {
            'y_true_all': [], 'y_pred_all': [], 'y_pred_proba_all': [],
            'param_combos1': [], 'param_combos2': [], 'accuracies': []
        }
    return local_results_all

def process_results(results, local_results_all):
    # Process and analyze results
    # This function would contain the code to aggregate results, compute metrics, etc.
    pass

if __name__ == "__main__":
    main()