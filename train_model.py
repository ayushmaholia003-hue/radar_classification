from src.ml_pipeline import MLRadarPipeline

def main():
    print("="*60)
    print("ML-BASED RADAR CLASSIFICATION SYSTEM")
    print("Training XGBoost Model")
    print("="*60)
    
    # Initialize pipeline
    pipeline = MLRadarPipeline()
    
    # Train on dataset
    train_acc, test_acc = pipeline.train('data/radar_dataset_with_noise.csv', test_size=0.25)
    
    # Save model
    pipeline.save_model(
        'models/xgboost_classifier.pkl',
        'models/preprocessor.pkl'
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Final Training Accuracy: {train_acc:.2%}")
    print(f"Final Test Accuracy: {test_acc:.2%}")
    print("\nModel saved to models/")


if __name__ == '__main__':
    main()
