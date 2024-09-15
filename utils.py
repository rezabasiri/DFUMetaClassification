from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, f1_score, recall_score, make_scorer

def apply_sampling(X, y, weights, oversampling_method='none', undersampling_method='none'):
    if oversampling_method != 'none':
        oversampling = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = oversampling.fit_resample(X, y)
        indices = oversampling.fit_resample(X, y)[1]
        weights_resampled = np.repeat(weights, [sum(indices == i) for i in range(len(X))])
        X, y, weights = X_resampled, y_resampled, weights_resampled
    if undersampling_method != 'none':
        undersampling = RandomUnderSampler(random_state=42)
        X, y, weights = undersampling.fit_resample(X, y, weights)
    return X, y, weights

def transform_labels(y, threshold):
    return (y > threshold).astype(int)

def custom_average_score(y_true, y_pred, **kwargs):
    y_pred = y_pred.reshape(-1).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    ordinal_weights = np.abs(y_true - y_pred)
    ordinal_weighted_accuracy = 1 - np.mean(ordinal_weights) / (len(np.unique(y_true)))
    f1 = f1_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    combined_metric = (accuracy + f1 + ordinal_weighted_accuracy) / 3
    return combined_metric

custom_scorer = make_scorer(custom_average_score)