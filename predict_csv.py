import sys
import pandas as pd
from src.ml_pipeline import MLRadarPipeline


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 predict_csv.py <input_csv_file>")
        print("Example: python3 predict_csv.py data/test_dataset.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    pipeline = MLRadarPipeline()
    pipeline.load_model(
        'models/xgboost_classifier.pkl',
        'models/preprocessor.pkl'
    )
    
    df = pd.read_csv(input_file)
    
    print(f"Predictions for {input_file}:")
    print("-" * 80)
    
    for idx, row in df.iterrows():
        sample = row.to_dict()
        result = pipeline.predict(sample)
        print(f"Sample {idx + 1}: {result}")
    
    print("-" * 80)
    print(f"Total samples processed: {len(df)}")


if __name__ == '__main__':
    main()
