from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from typing import Dict, Tuple
import joblib


class XGBoostRadarClassifier:

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        n_estimators: int = 200,
        random_state: int = 42
    ):
        """
        Initialize XGBoost classifier.
        
        Args:
            learning_rate: Learning rate (0.05-0.1 recommended)
            max_depth: Maximum tree depth (4-6 recommended)
            n_estimators: Number of boosting rounds (100-300 recommended)
            random_state: Random seed for reproducibility
        """
        self.model = GradientBoostingClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=random_state,
            verbose=0
        )
        self.is_trained = False
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Train the Gradient Boosting model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional, not used by sklearn GB)
            y_val: Validation labels (optional, not used by sklearn GB)
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict radar types.
        
        Args:
            X: Feature matrix
        
        Returns:
            Predicted class labels
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
        
        Returns:
            Probability matrix (n_samples, n_classes)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X)
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence scores.
        
        Args:
            X: Feature matrix
        
        Returns:
            predictions: Predicted class labels
            confidences: Confidence scores (max probability)
        """
        probas = self.predict_proba(X)
        predictions = np.argmax(probas, axis=1)
        confidences = np.max(probas, axis=1)
        
        return predictions, confidences
    
    def get_feature_importance(self, feature_names=None) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            feature_names: List of feature names
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        
        importance = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        return dict(zip(feature_names, importance))
    
    def save(self, path: str):
        """Save trained model to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model.")
        
        joblib.dump(self.model, path)
    
    def load(self, path: str):
        """Load trained model from disk."""
        self.model = joblib.load(path)
        self.is_trained = True
