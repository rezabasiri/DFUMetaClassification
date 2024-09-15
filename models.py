from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight

class OrdinalRandomForestClassifier(RandomForestRegressor):
    def __init__(self, n_classes=3, class_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes
        self.class_weight = class_weight

    def fit(self, X, y):
        y_continuous = y / (self.n_classes - 1)
        
        if self.class_weight:
            sample_weight = compute_sample_weight(class_weight=self.class_weight, y=y)
        else:
            sample_weight = None
        
        return super().fit(X, y_continuous, sample_weight=sample_weight)

    def predict(self, X):
        y_pred_continuous = super().predict(X)
        return np.round(y_pred_continuous * (self.n_classes - 1)).astype(int)

def get_classifiers(class_weight_dict):
    return {
        'Ordinal Random Forest': RandomForestClassifier(random_state=42, class_weight=class_weight_dict),
        'SVM': SVC(probability=True, random_state=42, class_weight=None),
        'XGBoost': XGBClassifier(random_state=42, tree_method='hist', class_weight=None, n_jobs=1, verbosity=0, silent=True),
        'Neural Network': MLPClassifier(random_state=42, early_stopping=True, max_iter=500),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=False, allow_writing_files=False, save_snapshot=False)
    }

def get_param_grids():
    return {
        'Ordinal Random Forest': {
            'n_estimators': [30, 100, 600, 800, 1000, 2000, 4000, 6000, 10000],
            'class_weight': ['balanced', 'balanced_subsample', None]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'degree': [2, 3],
            'gamma': ['scale', 'auto']
        },
        'XGBoost': {
            'max_depth': [6, 12],
            'learning_rate': [0.01, 0.3, 0.9],
            'n_estimators': [50, 100, 300]
        },
        'Neural Network': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
            'alpha': [0.001, 0.01],
            'learning_rate_init': [0.001, 0.1],
        },
        'CatBoost': {
            'iterations': [100, 500, 1000],
            'learning_rate': [0.01],
            'depth': [6, 12, 24],
            'l2_leaf_reg': [0, 0.1, 0.5],
        }
    }