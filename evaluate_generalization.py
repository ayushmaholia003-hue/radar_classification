
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.data_preprocessing import RadarDataPreprocessor
from src.ml_classifier import XGBoostRadarClassifier

def evaluate_generalization():
    df = pd.read_csv('data/radar_dataset_with_noise.csv')
    preprocessor = RadarDataPreprocessor()
    X, y = preprocessor.prepare_features(df, fit=True)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    classifier = XGBoostRadarClassifier(
        learning_rate=0.05,
        max_depth=3,
        n_estimators=100,
        random_state=42
    )
    
    classifier.train(X_train, y_train, X_val, y_val)
    
    y_train_pred, conf_train = classifier.predict_with_confidence(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    
    y_val_pred, conf_val = classifier.predict_with_confidence(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    y_test_pred, conf_test = classifier.predict_with_confidence(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    train_val_gap = train_acc - val_acc
    val_test_gap = abs(val_acc - test_acc)
    
    print("\n" + "="*60)
    print("MODEL EVALUATION (60-20-20 Split)")
    print("="*60)
    print(f"Train Accuracy:       {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Validation Accuracy:  {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"Test Accuracy:        {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"\nTrain-Val Gap:        {train_val_gap:+.4f} ({train_val_gap*100:+.2f}%)")
    print(f"Val-Test Gap:         {val_test_gap:+.4f} ({val_test_gap*100:+.2f}%)")
    
    print(f"\nConfidence Means:")
    print(f"  Train:  {conf_train.mean():.4f}")
    print(f"  Val:    {conf_val.mean():.4f}")
    print(f"  Test:   {conf_test.mean():.4f}")
    
    print(f"\nOverfitted predictions (>=0.99):")
    print(f"  Train: {(conf_train >= 0.99).sum()/len(conf_train)*100:.1f}%")
    print(f"  Val:   {(conf_val >= 0.99).sum()/len(conf_val)*100:.1f}%")
    print(f"  Test:  {(conf_test >= 0.99).sum()/len(conf_test)*100:.1f}%")
    
    
    results = {
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'test_accuracy': test_acc,
        'train_confidence_mean': conf_train.mean(),
        'val_confidence_mean': conf_val.mean(),
        'test_confidence_mean': conf_test.mean(),
        'train_val_gap': train_val_gap,
        'val_test_gap': val_test_gap,
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv('evaluation_results.csv', index=False)


if __name__ == '__main__':
    evaluate_generalization()
