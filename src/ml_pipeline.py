import pandas as pd
import numpy as np
from typing import List, Dict
import joblib

from .data_preprocessing import RadarDataPreprocessor
from .ml_classifier import XGBoostRadarClassifier


class MLRadarPipeline:
    
    def __init__(self, model_path: str = None):
        """
        Initialize ML pipeline.
        
        Args:
            model_path: Path to saved model (optional)
        """
        self.preprocessor = RadarDataPreprocessor()
        self.classifier = XGBoostRadarClassifier()
        
        if model_path:
            self.load_model(model_path)
    
    def train(self, csv_path: str, test_size: float = 0.2):
        """
        Train the ML pipeline on dataset.
        
        Args:
            csv_path: Path to CSV dataset
            test_size: Fraction of data for testing
        """
        # Load data
        df = pd.read_csv(csv_path)
        
        print(f"Loaded {len(df)} samples")
        print(f"Classes: {df['radar_name'].unique()}")
        print(f"Class distribution:\n{df['radar_name'].value_counts()}")
        
        # Prepare features
        X, y = self.preprocessor.prepare_features(df, fit=True)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train model
        print("\nTraining XGBoost model...")
        self.classifier.train(X_train, y_train, X_test, y_test)
        
        # Evaluate
        train_acc = self.classifier.model.score(X_train, y_train)
        test_acc = self.classifier.model.score(X_test, y_test)
        
        print(f"\nTraining accuracy: {train_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Feature importance
        importance = self.classifier.get_feature_importance(
            self.preprocessor.get_feature_names()
        )
        print("\nTop 5 important features:")
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_features[:5]:
            print(f"  {feat}: {imp:.4f}")
        
        return train_acc, test_acc
    
    def predict(self, data: Dict) -> Dict[str, any]:
        """
        Predict radar type for single sample.
        
        Args:
            data: Dictionary with feature values
        
        Returns:
            {
                "radar_name": str,
                "confidence": float
            }
        """
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess
        X, _ = self.preprocessor.prepare_features(df, fit=False)
        
        # Predict
        predictions, confidences = self.classifier.predict_with_confidence(X)
        
        # Decode label
        radar_name = self.preprocessor.get_label_name(predictions[0])
        confidence = float(confidences[0])
        
        return {
            "radar_name": radar_name,
            "confidence": confidence
        }
    
    def predict_batch(self, csv_path: str) -> List[Dict[str, any]]:
        """
        Predict radar types for batch of samples.
        
        Args:
            csv_path: Path to CSV file
        
        Returns:
            List of prediction dictionaries
        """
        df = pd.read_csv(csv_path)
        
        # Preprocess
        X, _ = self.preprocessor.prepare_features(df, fit=False)
        
        # Predict
        predictions, confidences = self.classifier.predict_with_confidence(X)
        
        # Format results
        results = []
        for pred, conf in zip(predictions, confidences):
            radar_name = self.preprocessor.get_label_name(pred)
            results.append({
                "radar_name": radar_name,
                "confidence": float(conf)
            })
        
        return results
    
    def save_model(self, model_path: str, preprocessor_path: str):
        """
        Save trained model and preprocessor.
        
        Args:
            model_path: Path to save model
            preprocessor_path: Path to save preprocessor
        """
        self.classifier.save(model_path)
        joblib.dump(self.preprocessor, preprocessor_path)
        print(f"Model saved to {model_path}")
        print(f"Preprocessor saved to {preprocessor_path}")
    
    def load_model(self, model_path: str, preprocessor_path: str = None):
        """
        Load trained model and preprocessor.
        
        Args:
            model_path: Path to model file
            preprocessor_path: Path to preprocessor file
        """
        self.classifier.load(model_path)
        
        if preprocessor_path:
            self.preprocessor = joblib.load(preprocessor_path)
        
        print(f"Model loaded from {model_path}")
