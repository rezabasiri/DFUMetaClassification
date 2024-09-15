import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score

from utils import apply_sampling, transform_labels, custom_scorer
from models import get_classifiers, get_param_grids
from feature_engineering import calculate_shap_values

def process_patient(patient, data, target_class, n_features, selected_methods):
    # Split the data
    test_data = data[data['Patient#'] == patient]
    train_data = data[data['Patient#'] != patient]

    # Prepare features and labels
    X_train = train_data.drop(columns=[target_class, 'Patient#'])
    y_train = train_data[target_class]
    X_test = test_data.drop(columns=[target_class, 'Patient#'])
    y_test = test_data[target_class]

    # Impute missing values
    imputer = KNNImputer(n_neighbors=5)
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # Apply one-hot encoding
    onehot_columns = ['Foot Aspect', 'Type of Pain Grouped']
    onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    onehot_encoder.fit(X_train[onehot_columns])
    
    X_train_encoded = pd.DataFrame(onehot_encoder.transform(X_train[onehot_columns]), 
                                   columns=onehot_encoder.get_feature_names_out(onehot_columns), 
                                   index=X_train.index)
    X_test_encoded = pd.DataFrame(onehot_encoder.transform(X_test[onehot_columns]), 
                                  columns=onehot_encoder.get_feature_names_out(onehot_columns), 
                                  index=X_test.index)
    
    X_train = pd.concat([X_train.drop(columns=onehot_columns), X_train_encoded], axis=1)
    X_test = pd.concat([X_test.drop(columns=onehot_columns), X_test_encoded], axis=1)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    # Apply oversampling
    X_train_scaled, y_train, _ = apply_sampling(X_train_scaled, y_train, None, oversampling_method='random', undersampling_method='none')

    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    # Get classifiers and parameter grids
    classifiers = get_classifiers(class_weight_dict)
    param_grids = get_param_grids()

    results = {}

    for clf_config in selected_methods:
        if isinstance(clf_config, dict):
            results.update(ensemble_approach(clf_config, classifiers, param_grids, X_train_scaled, y_train, X_test_scaled, y_test))
        else:
            results.update(single_classifier_approach(clf_config, classifiers, param_grids, X_train_scaled, y_train, X_test_scaled, y_test, n_features))

    return results

def ensemble_approach(clf_config, classifiers, param_grids, X_train, y_train, X_test, y_test):
    ensemble_results = []
    for config_name, config in clf_config.items():
        clf1_name = clf2_name = config[0] if len(config) == 1 else config
        clf1 = classifiers[clf1_name]
        clf2 = classifiers[clf2_name] if isinstance(clf2_name, str) else classifiers[clf2_name[1]]

        param_grid1 = param_grids[clf1_name]
        param_grid2 = param_grids[clf2_name] if isinstance(clf2_name, str) else param_grids[clf2_name[1]]

        y_train_bin1 = transform_labels(y_train, 0)
        clf1 = GridSearchCV(clf1, param_grid1, scoring=custom_scorer, n_jobs=-1, verbose=0, cv=2, refit=True)
        clf1.fit(X_train, y_train_bin1)
        best_params1 = clf1.best_params_

        y_train_bin2 = transform_labels(y_train, 1)
        clf2 = GridSearchCV(clf2, param_grid2, scoring=custom_scorer, n_jobs=-1, verbose=0, cv=2, refit=True)
        clf2.fit(X_train, y_train_bin2)
        best_params2 = clf2.best_params_

        test_prob1 = clf1.predict_proba(X_test)[:, 1]
        test_prob2 = clf2.predict_proba(X_test)[:, 1]

        prob_I = 1 - test_prob1
        prob_P = test_prob1 * (1 - test_prob2)
        prob_R = test_prob2

        ensemble_results.append((prob_I, prob_P, prob_R))

    prob_I_ensemble = np.mean([result[0] for result in ensemble_results], axis=0)
    prob_P_ensemble = np.mean([result[1] for result in ensemble_results], axis=0)
    prob_R_ensemble = np.mean([result[2] for result in ensemble_results], axis=0)

    y_pred = np.argmax(np.vstack([prob_I_ensemble, prob_P_ensemble, prob_R_ensemble]), axis=0)
    
    clf_names = [f"{config_name}_{'_'.join(clf_config[config_name])}" for config_name in clf_config]
    clf_name = '_'.join(clf_names)
    return {clf_name: {'y_true': y_test, 'y_pred': y_pred, 'best_params1': best_params1, 'best_params2': best_params2}}

def single_classifier_approach(clf_config, classifiers, param_grids, X_train, y_train, X_test, y_test, n_features):
    clf1_name = clf2_name = clf_config[0] if len(clf_config) == 1 else clf_config
    clf1t = classifiers[clf1_name]
    clf2t = classifiers[clf2_name]
    param_grid1 = param_grids[clf1_name]
    param_grid2 = param_grids[clf2_name]

    # Calculate SHAP values and select top features
    shap_values = calculate_shap_values(X_train, y_train)
    feature_importance = np.sum(np.abs(shap_values), axis=0)
    top_indices = np.argsort(feature_importance)[-n_features:]
    top_indices = np.sort(top_indices)

    X_train_top = X_train.iloc[:, top_indices]
    X_test_top = X_test.iloc[:, top_indices]

    y_train_bin1 = transform_labels(y_train, 0)
    clf1 = GridSearchCV(clf1t, param_grid1, scoring=custom_scorer, n_jobs=1, verbose=0, cv=2, refit=True)
    clf1.fit(X_train_top, y_train_bin1)
    best_params1 = clf1.best_params_

    y_train_bin2 = transform_labels(y_train, 1)
    clf2 = GridSearchCV(clf2t, param_grid2, scoring=custom_scorer, n_jobs=1, verbose=0, cv=2, refit=True)
    clf2.fit(X_train_top, y_train_bin2)
    best_params2 = clf2.best_params_

    test_prob1 = clf1.predict_proba(X_test_top)[:, 1]
    test_prob2 = clf2.predict_proba(X_test_top)[:, 1]

    prob_I = 1 - test_prob1
    prob_P = test_prob1 * (1 - test_prob2)
    prob_R = test_prob2
    
    y_pred_proba = np.column_stack((prob_I, prob_P, prob_R))
    y_pred = np.argmax(np.vstack([prob_I, prob_P, prob_R]), axis=0)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    clf_name = '_'.join(clf_config) if isinstance(clf_config, list) else clf_config
    return {clf_name: {'y_true': y_test, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba, 
                       'best_params1': best_params1, 'best_params2': best_params2, 'accuracies': accuracy}}