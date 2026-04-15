from src.ml_pipeline import MLRadarPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np

def main():
    print("="*60)
    print("ML-BASED RADAR CLASSIFICATION SYSTEM")
    print("Training on Realistic Dataset")
    print("="*60)
    
    pipeline = MLRadarPipeline()
    
    train_acc, test_acc = pipeline.train('data/radar_dataset_with_noise.csv', test_size=0.25)
    
    overfitting_gap = train_acc - test_acc
    
    df = pd.read_csv('data/radar_dataset_with_noise.csv')
    X, y = pipeline.preprocessor.prepare_features(df, fit=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    _, conf_train = pipeline.classifier.predict_with_confidence(X_train)
    _, conf_test = pipeline.classifier.predict_with_confidence(X_test)
    
    pipeline.save_model(
        'models/xgboost_classifier.pkl',
        'models/preprocessor.pkl'
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Final Training Accuracy: {train_acc:.4f} ({train_acc:.2%})")
    print(f"Final Test Accuracy:     {test_acc:.4f} ({test_acc:.2%})")
    print(f"Overfitting Gap:         {overfitting_gap:.4f} ({overfitting_gap:.2%})")
    
    print("\n" + "-"*60)
    print("CONFIDENCE SCORE ANALYSIS:")
    print("-"*60)
    print(f"Train - Min: {conf_train.min():.4f}, Mean: {conf_train.mean():.4f}, Max: {conf_train.max():.4f}")
    print(f"Test  - Min: {conf_test.min():.4f}, Mean: {conf_test.mean():.4f}, Max: {conf_test.max():.4f}")
    
    high_conf_test = (conf_test >= 0.99).sum() / len(conf_test) * 100
    low_conf_test = (conf_test < 0.70).sum() / len(conf_test) * 100
    
    print(f"\nConfidence distribution on test set:")
    print(f"  >= 0.99 (overconfident):  {high_conf_test:.1f}%")
    print(f"  < 0.70 (uncertain):       {low_conf_test:.1f}%")
    print(f"  0.70-0.99 (confident):    {100 - high_conf_test - low_conf_test:.1f}%")


if __name__ == '__main__':
    main()
