import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path, target_class):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Preprocess the data
    data_filtered = preprocess_data(data, target_class)
    
    return data_filtered

def preprocess_data(data, target_class):
    # Drop unnecessary columns
    columns_to_drop = ['Depth_Images', 'Thermal_Images', 'ID', 'Location', 'DFU#', 'Appt#', 'Appt Days']
    data_filtered = data.drop(columns=columns_to_drop)
    
    # Perform label encoding
    data_filtered, _ = label_encode_columns(data_filtered)
    
    # Map categorical variables
    data_filtered = map_categorical_variables(data_filtered)
    
    return data_filtered

def label_encode_columns(data):
    text_columns = ['Sex (F:0, M:1)', 'Side (Left:0, Right:1)', 'Foot Aspect', 'Odor', 'Type of Pain Grouped']
    label_encoders = {}
    
    for column in text_columns:
        if column in data.columns:
            mask = data[column].notna()
            label_encoders[column] = LabelEncoder()
            label_encoders[column].fit(data.loc[mask, column].astype(str))
            data[f'{column}_encoded'] = data[column].astype(str)
            data.loc[mask, f'{column}_encoded'] = label_encoders[column].transform(data.loc[mask, column].astype(str))
            data[column] = data[f'{column}_encoded']
            data = data.drop(columns=[f'{column}_encoded'])
    
    return data, label_encoders

def map_categorical_variables(data):
    categorical_mappings = {
        'Healing Phase': {'I': 0, 'P': 1, 'R': 2},
        'Location Grouped (Hallux:1,Toes,Middle,Heel,Ankle:5)': {'ankle': 4, 'Heel': 3, 'middle': 2, 'toes': 1, 'Hallux': 0},
        'Dressing Grouped': {'NoDressing': 0, 'BandAid': 1, 'BasicDressing': 1, 'AbsorbantDressing': 2, 'Antiseptic': 3, 'AdvanceMethod': 4, 'other': 4},
        'Exudate Appearance (Serous:1,Haemoserous,Bloody,Thick:4)': {'Serous': 0, 'Haemoserous': 1, 'Bloody': 2, 'Thick': 3}
    }
    
    for column, mapping in categorical_mappings.items():
        data[column] = data[column].map(mapping)
    
    return data.astype(float)