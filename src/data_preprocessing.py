import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple

class RadarDataPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.feature_names = ['PRI', 'PW', 'FREQUENCY']
        
    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features from 3-feature dataset: PRI, PW, FREQUENCY
        Returns X (shape: n_samples, 3), y (encoded labels)
        """
        df = df.copy()
        
        # Handle missing values with median
        numeric_cols = self.feature_names
        for col in numeric_cols:
            if col in df.columns:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        # Select features
        X = df[self.feature_names].values
        
        y = None
        if 'radar_name' in df.columns:
            if fit:
                y = self.label_encoder.fit_transform(df['radar_name'])
            else:
                y = self.label_encoder.transform(df['radar_name'])
        
        return X, y
    
    def get_label_name(self, encoded_label: int) -> str:
        """Get class name from encoded label"""
        return self.label_encoder.inverse_transform([encoded_label])[0]
    
    def get_feature_names(self):
        """Return feature names"""
        return self.feature_names
