"""
Data Preprocessing Module
Handles encoding, feature engineering, and data preparation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple


class RadarDataPreprocessor:
    """Preprocesses radar signal data for ML training."""
    
    def __init__(self):
        self.pri_pattern_encoder = LabelEncoder()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical PRI_pattern feature."""
        df = df.copy()
        
        # Encode PRI_pattern: constant=0, jittered=1, staggered=2
        df['PRI_pattern_encoded'] = self.pri_pattern_encoder.fit_transform(df['PRI_pattern'])
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate derived features for better classification.
        
        Derived features:
        - PRI_stability: Inverse of PRI variance (higher = more stable)
        - frequency_stability: Inverse of frequency variance
        - pulse_energy: Approximation based on pulse width and duty cycle
        - frequency_band: Categorized frequency ranges
        """
        df = df.copy()
        
        # Stability features (handle division by zero)
        df['PRI_stability'] = 1.0 / (1.0 + df['PRI_variance'])
        df['frequency_stability'] = 1.0 / (1.0 + df['frequency_variance'])
        
        # Pulse energy approximation
        df['pulse_energy'] = df['mean_pulse_width'] * df['duty_cycle']
        
        # Frequency band categorization
        df['frequency_band'] = pd.cut(
            df['mean_frequency'],
            bins=[0, 2000, 4000, 6000, 8000, 12000],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)
        
        # PRI range categorization
        df['PRI_range'] = pd.cut(
            df['mean_PRI'],
            bins=[0, 500, 1000, 2000, 6000],
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        # Interaction features
        df['PRI_freq_ratio'] = df['mean_PRI'] / (df['mean_frequency'] + 1)
        df['variance_ratio'] = df['PRI_variance'] / (df['frequency_variance'] + 1)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature matrix and labels.
        
        Args:
            df: Input dataframe
            fit: Whether to fit encoders (True for training, False for inference)
        
        Returns:
            X: Feature matrix
            y: Encoded labels (None if 'radar_name' not in df)
        """
        df = df.copy()
        
        # Encode categorical features
        if fit:
            df = self.encode_categorical(df)
        else:
            df['PRI_pattern_encoded'] = self.pri_pattern_encoder.transform(df['PRI_pattern'])
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Select features for model
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
        
        # Encode labels if present
        y = None
        if 'radar_name' in df.columns:
            if fit:
                y = self.label_encoder.fit_transform(df['radar_name'])
            else:
                y = self.label_encoder.transform(df['radar_name'])
        
        return X, y
    
    def get_label_name(self, encoded_label: int) -> str:
        """Convert encoded label back to radar name."""
        return self.label_encoder.inverse_transform([encoded_label])[0]
    
    def get_feature_names(self):
        """Return list of feature names."""
        return self.feature_names
