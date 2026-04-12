from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from typing import Dict, Tuple
import joblib

class XGBoostRadarClassifier:

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        self.base_model = GradientBoostingClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=random_state,
            subsample=0.8, 
            min_samples_split=10,  
            min_samples_leaf=5,  
            verbose=0
        )
        self.model = None 
        self.is_trained = False
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None):

        self.base_model.fit(X_train, y_train)
        
        self.model = CalibratedClassifierCV(
            self.base_model, 
            method='sigmoid',
            cv='prefit'
        )
        
        if X_val is not None and y_val is not None:
            self.model.fit(X_val, y_val)
        else:
            self.model.fit(X_train, y_train)
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:

        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X)
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        probas = self.predict_proba(X)
        predictions = np.argmax(probas, axis=1)
        confidences = np.max(probas, axis=1)
        
        return predictions, confidences
    
    def get_feature_importance(self, feature_names=None) -> Dict[str, float]:

        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        
        importance = self.base_model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        return dict(zip(feature_names, importance))
    
    def save(self, path: str):

        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model.")
        
        joblib.dump(self.model, path)
    
    def load(self, path: str):

        self.model = joblib.load(path)
        self.is_trained = True
