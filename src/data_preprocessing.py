import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple

class RadarDataPreprocessor:
    def __init__(self):
        self.pri_pattern_encoder = LabelEncoder()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Encode PRI_pattern: constant=0, jittered=1, staggered=2
        df['PRI_pattern_encoded'] = self.pri_pattern_encoder.fit_transform(df['PRI_pattern'])
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['PRI_stability'] = 1.0 / (1.0 + df['PRI_variance'])
        df['frequency_stability'] = 1.0 / (1.0 + df['frequency_variance'])
        
        df['pulse_energy'] = df['mean_pulse_width'] * df['duty_cycle']
        
        df['frequency_band'] = pd.cut(
            df['mean_frequency'],
            bins=[0, 2000, 4000, 6000, 8000, 12000],
            labels=[0, 1, 2, 3, 4],
            include_lowest=True
        )
        df['frequency_band'] = df['frequency_band'].cat.codes
        
        df['PRI_range'] = pd.cut(
            df['mean_PRI'],
            bins=[0, 500, 1000, 2000, 6000],
            labels=[0, 1, 2, 3],
            include_lowest=True
        )
        df['PRI_range'] = df['PRI_range'].cat.codes
        
        df['PRI_freq_ratio'] = df['mean_PRI'] / (df['mean_frequency'] + 1)
        df['variance_ratio'] = df['PRI_variance'] / (df['frequency_variance'] + 1)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:

        df = df.copy()
        
        numeric_cols = ['mean_PRI', 'PRI_variance', 'mean_frequency', 
                       'frequency_variance', 'mean_pulse_width', 'duty_cycle']
        for col in numeric_cols:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median() if fit else 0
                df[col] = df[col].fillna(median_val)
        
        if 'PRI_pattern' in df.columns and df['PRI_pattern'].isnull().any():
            mode_val = df['PRI_pattern'].mode()[0] if fit and len(df['PRI_pattern'].mode()) > 0 else 'constant'
            df['PRI_pattern'] = df['PRI_pattern'].fillna(mode_val)
        
        if fit:
            df = self.encode_categorical(df)
        else:
            df['PRI_pattern_encoded'] = self.pri_pattern_encoder.transform(df['PRI_pattern'])
        
        df = self.engineer_features(df)
        
        feature_columns = [
            'mean_PRI',
            'PRI_variance',
            'PRI_pattern_encoded',
            'mean_frequency',
            'frequency_variance',
            'mean_pulse_width',
            'duty_cycle',
            'PRI_stability',
            'frequency_stability',
            'pulse_energy',
            'frequency_band',
            'PRI_range',
            'PRI_freq_ratio',
            'variance_ratio'
        ]
        
        self.feature_names = feature_columns
        X = df[feature_columns].values
        
        y = None
        if 'radar_name' in df.columns:
            if fit:
                y = self.label_encoder.fit_transform(df['radar_name'])
            else:
                y = self.label_encoder.transform(df['radar_name'])
        
        return X, y
    
    def get_label_name(self, encoded_label: int) -> str:
        return self.label_encoder.inverse_transform([encoded_label])[0]
    
    def get_feature_names(self):
        return self.feature_names
