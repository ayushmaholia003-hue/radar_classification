import pandas as pd
import numpy as np
import random


def sample_normal(mean, std, min_val, max_val):
    """Sample from normal distribution with bounds"""
    val = np.random.normal(mean, std)
    return max(min_val, min(max_val, val))


def generate_realistic_samples(n, label, params, noise_prob=0.2):
    """Generate realistic radar samples with overlapping distributions"""
    data = []

    for _ in range(n):
        # Use normal distributions (realistic spread)
        mean_PRI = sample_normal(*params["pri"])
        pri_var = sample_normal(*params["pri_var"])
        mean_freq = sample_normal(*params["freq"])
        freq_var = sample_normal(*params["freq_var"])
        pulse_width = sample_normal(*params["pw"])
        
        # IMPORTANT: Derived feature (duty cycle depends on pulse width and PRI)
        duty = pulse_width / mean_PRI
        
        # Add correlated multiplicative noise (more realistic than additive)
        if random.random() < noise_prob:
            mean_PRI *= np.random.uniform(0.8, 1.2)
            mean_freq *= np.random.uniform(0.9, 1.1)
            pulse_width *= np.random.uniform(0.8, 1.2)
            duty = pulse_width / mean_PRI  # Recalculate after noise

        # Randomize PRI pattern (no fixed mapping per class)
        pattern = random.choice(["constant", "staggered", "jittered"])

        row = {
            "mean_PRI": round(mean_PRI),
            "PRI_variance": round(pri_var),
            "PRI_pattern": pattern,
            "mean_frequency": round(mean_freq),
            "frequency_variance": round(freq_var),
            "mean_pulse_width": round(pulse_width, 2),
            "duty_cycle": round(duty, 5),
            "radar_name": label
        }

        data.append(row)

    return data


# OVERLAPPING PARAMETER DISTRIBUTIONS
# (mean, std, min, max) for each feature
params = {
    "search_radar": {
        "pri": (1800, 300, 1200, 2500),
        "pri_var": (150, 50, 50, 300),
        "freq": (3100, 400, 2500, 4000),
        "freq_var": (60, 20, 30, 120),
        "pw": (5.5, 1.0, 3.5, 7.5)
    },
    "fire_control_radar": {
        "pri": (350, 100, 150, 700),
        "pri_var": (20, 10, 5, 50),
        "freq": (9500, 800, 7000, 11000),
        "freq_var": (30, 10, 10, 60),
        "pw": (0.9, 0.3, 0.4, 1.8)
    },
    "tracking_radar": {
        "pri": (700, 200, 400, 1200),
        "pri_var": (12, 6, 5, 40),
        "freq": (8500, 700, 7000, 10000),
        "freq_var": (20, 10, 10, 50),
        "pw": (1.5, 0.5, 0.8, 2.5)
    },
    "weather_radar": {
        "pri": (4500, 600, 3000, 6000),
        "pri_var": (300, 100, 100, 500),
        "freq": (3000, 300, 2500, 3500),
        "freq_var": (110, 30, 60, 160),
        "pw": (8.5, 1.5, 6.0, 11.0)
    },
    "air_traffic_control": {
        "pri": (1000, 200, 700, 1500),
        "pri_var": (80, 20, 40, 120),
        "freq": (1200, 200, 800, 1800),
        "freq_var": (40, 10, 20, 70),
        "pw": (3.0, 0.5, 2.0, 4.0)
    },
    "maritime_radar": {
        "pri": (1500, 300, 1000, 2000),
        "pri_var": (120, 30, 60, 180),
        "freq": (5000, 600, 3500, 6500),
        "freq_var": (70, 15, 40, 100),
        "pw": (4.5, 0.7, 3.0, 6.0)
    }
}

# Generate dataset with overlapping, realistic distributions
dataset = []

for label, param in params.items():
    dataset += generate_realistic_samples(200, label, param, noise_prob=0.2)

df = pd.DataFrame(dataset)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("data/radar_dataset_with_noise.csv", index=False)

print("="*60)
print("REALISTIC RADAR DATASET GENERATION")
print("="*60)
print(f"Dataset generated: {df.shape}")
print(f"Total samples: {len(df)}")
print(f"\nClass distribution:")
print(df['radar_name'].value_counts())
print(f"\nFeature statistics:")
print(df[['mean_PRI', 'mean_frequency', 'mean_pulse_width', 'duty_cycle']].describe())
print(f"\nDataset features:")
print(f"  - Normal distributions (realistic spread)")
print(f"  - Overlapping ranges across classes")
print(f"  - Derived duty_cycle (pulse_width/mean_PRI)")
print(f"  - Multiplicative noise ({20}% samples)")
print(f"  - Randomized PRI patterns")
print("="*60)
