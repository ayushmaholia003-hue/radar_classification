import pandas as pd
import numpy as np
import random
import os

def sample_normal(mean, std, min_val, max_val):
    # Sample from normal distribution with bounds
    val = np.random.normal(mean, std)
    return max(min_val, min(max_val, val))

def generate_realistic_samples(n, label, params, noise_prob=0.2):
    # Generate realistic radar samples with only PRI, PW, FREQUENCY
    # Overlapping distributions across classes
    data = []

    for _ in range(n):
        PRI = sample_normal(*params["pri"])
        FREQUENCY = sample_normal(*params["freq"])
        PW = sample_normal(*params["pw"])

        # Add correlated multiplicative noise
        if random.random() < noise_prob:
            PRI *= np.random.uniform(0.8, 1.2)
            FREQUENCY *= np.random.uniform(0.9, 1.1)
            PW *= np.random.uniform(0.8, 1.2)

        # Ensure bounds after noise
        PRI = max(params["pri"][2], min(params["pri"][3], PRI))
        FREQUENCY = max(params["freq"][2], min(params["freq"][3], FREQUENCY))
        PW = max(params["pw"][2], min(params["pw"][3], PW))

        row = {
            "PRI": round(PRI),
            "PW": round(PW, 2),
            "FREQUENCY": round(FREQUENCY),
            "radar_name": label
        }

        data.append(row)

    return data

# Realistic params (mean, std, min, max) for PRI, PW, FREQUENCY per class
# Adapted from original distributions
params = {
    "search_radar": {
        "pri": (1800, 300, 1200, 2500),
        "freq": (3100, 400, 2500, 4000),
        "pw": (5.5, 1.0, 3.5, 7.5)
    },
    "fire_control_radar": {
        "pri": (350, 100, 150, 700),
        "freq": (9500, 800, 7000, 11000),
        "pw": (0.9, 0.3, 0.4, 1.8)
    },
    "tracking_radar": {
        "pri": (700, 200, 400, 1200),
        "freq": (8500, 700, 7000, 10000),
        "pw": (1.5, 0.5, 0.8, 2.5)
    },
    "weather_radar": {
        "pri": (4500, 600, 3000, 6000),
        "freq": (3000, 300, 2500, 3500),
        "pw": (8.5, 1.5, 6.0, 11.0)
    },
    "air_traffic_control": {
        "pri": (1000, 200, 700, 1500),
        "freq": (1200, 200, 800, 1800),
        "pw": (3.0, 0.5, 2.0, 4.0)
    },
    "maritime_radar": {
        "pri": (1500, 300, 1000, 2000),
        "freq": (5000, 600, 3500, 6500),
        "pw": (4.5, 0.7, 3.0, 6.0)
    }
}

# Generate balanced dataset
print("Generating 3-feature radar dataset...")
dataset = []

for label, param in params.items():
    samples = generate_realistic_samples(200, label, param, noise_prob=0.2)
    dataset.extend(samples)

df = pd.DataFrame(dataset)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Save full dataset
df.to_csv("data/radar_dataset_with_noise.csv", index=False)

# Save small sample for radar.csv (first 15 rows)
sample_df = df.head(15)
sample_df.to_csv("radar.csv", index=False)

print("="*70)
print("3-FEATURE RADAR DATASET GENERATED")
print("="*70)
print(f"Dataset shape: {df.shape}")
print(f"Classes: {len(df['radar_name'].unique())}")
print("\nClass distribution:")
print(df['radar_name'].value_counts().sort_index())
print("\nFeature statistics:")
print(df[['PRI', 'PW', 'FREQUENCY']].describe())
print("\nSample data:")
print(df.head())
print("\nFeatures:")
print("  - PRI (Pulse Repetition Interval, µs)")
print("  - PW (Pulse Width, µs)") 
print("  - FREQUENCY (MHz)")
print("  - Realistic overlapping distributions")
print("  - Multiplicative noise (20% samples)")
print("  - Same class parameter ranges as original")
print("="*70)
