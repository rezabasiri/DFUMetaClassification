import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import shap
from models import OrdinalRandomForestClassifier

def engineer_features(data):
    data['BMI'] = data['Weight (Kg)'] / ((data['Height (cm)'] / 100) ** 2)
    data['Age above 60'] = (data['Age'] > 60).astype(int)
    data['Age Bin'] = pd.cut(data['Age'], bins=range(0, int(data['Age'].max()) + 20, 20), right=False, 
                             labels=range(len(range(0, int(data['Age'].max()) + 20, 20)) - 1))
    data['Weight Bin'] = pd.cut(data['Weight (Kg)'], bins=range(0, int(data['Weight (Kg)'].max()) + 20, 20), right=False, 
                                labels=range(len(range(0, int(data['Weight (Kg)'].max()) + 20, 20)) - 1))
    data['Height Bin'] = pd.cut(data['Height (cm)'], bins=range(0, int(data['Height (cm)'].max()) + 10, 10), right=False, 
                                labels=range(len(range(0, int(data['Height (cm)'].max()) + 10, 10)) - 1))
    return data

def calculate_shap_values(X_train_scaled, y_train):
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    # clf0 = RandomForestClassifier(random_state=42, class_weight=class_weight_dict, n_estimators=1000, n_jobs=-1)
    clf0 = OrdinalRandomForestClassifier(n_estimators=1000, random_state=42, n_classes=3, class_weight=class_weight_dict, n_jobs=-1)
    clf0.fit(X_train_scaled, y_train)
    
    explainer = shap.TreeExplainer(clf0)
    shap_values = explainer.shap_values(X_train_scaled)
    
    return shap_values

def select_top_features(shap_values, feature_names, n_features=22):
    feature_importance = np.sum(np.abs(shap_values), axis=0)
    features_indices = np.argsort(feature_importance)[-n_features:]
    features_indices = np.sort(features_indices)
    return features_indices, [feature_names[i] for i in features_indices]