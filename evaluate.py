import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.ml_pipeline import MLRadarPipeline


def main():
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Load model
    pipeline = MLRadarPipeline()
    pipeline.load_model(
        'models/xgboost_classifier.pkl',
        'models/preprocessor.pkl'
    )
    
    df = pd.read_csv('data/radar_dataset_with_noise.csv')
    
    X, y_true = pipeline.preprocessor.prepare_features(df, fit=False)
    
    y_pred, confidences = pipeline.classifier.predict_with_confidence(X)
    
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Average Confidence: {np.mean(confidences):.4f}")
    print(f"Min Confidence: {np.min(confidences):.4f}")
    print(f"Max Confidence: {np.max(confidences):.4f}")
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    
    target_names = pipeline.preprocessor.label_encoder.classes_
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\nPredicted →")
    print("Actual ↓")
    print("\t" + "\t".join([name[:10] for name in target_names]))
    for i, name in enumerate(target_names):
        print(f"{name[:10]}\t" + "\t".join([str(cm[i, j]) for j in range(len(target_names))]))
    
    print("\n" + "="*60)
    print("PER-CLASS CONFIDENCE")
    print("="*60)
    
    for i, class_name in enumerate(target_names):
        class_mask = y_true == i
        class_confidences = confidences[class_mask]
        if len(class_confidences) > 0:
            print(f"{class_name:25s}: {np.mean(class_confidences):.4f} (±{np.std(class_confidences):.4f})")


if __name__ == '__main__':
    main()
