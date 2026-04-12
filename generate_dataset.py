"""
Generate realistic radar dataset with noise and overlapping characteristics.
"""
import pandas as pd
import numpy as np
import random


def generate_samples(n, label, pri_range, pri_var_range, freq_range,
                     freq_var_range, pw_range, duty_range, patterns, noise_level=0.15):
    data = []
    for i in range(n):
        mean_pri = np.random.randint(*pri_range)
        pri_var = np.random.randint(*pri_var_range)
        mean_freq = np.random.randint(*freq_range)
        freq_var = np.random.randint(*freq_var_range)
        pulse_width = round(np.random.uniform(*pw_range), 2)
        duty = round(np.random.uniform(*duty_range), 4)
        
        if random.random() < noise_level:
            mean_pri += int(np.random.normal(0, pri_range[1] * 0.1))
            pri_var += int(np.random.normal(0, pri_var_range[1] * 0.2))
            mean_freq += int(np.random.normal(0, freq_range[1] * 0.05))
            freq_var += int(np.random.normal(0, freq_var_range[1] * 0.2))
            pulse_width += round(np.random.normal(0, pw_range[1] * 0.1), 2)
            duty += round(np.random.normal(0, duty_range[1] * 0.1), 4)
            
            mean_pri = max(10, mean_pri)
            pri_var = max(1, pri_var)
            mean_freq = max(100, mean_freq)
            freq_var = max(1, freq_var)
            pulse_width = max(0.1, pulse_width)
            duty = max(0.0001, min(0.01, duty))
        
        row = {
            "mean_PRI": mean_pri,
            "PRI_variance": pri_var,
            "PRI_pattern": random.choice(patterns),
            "mean_frequency": mean_freq,
            "frequency_variance": freq_var,
            "mean_pulse_width": pulse_width,
            "duty_cycle": duty,
            "radar_name": label
        }
        data.append(row)
    return data


def add_edge_cases(dataset):
    edge_cases = [
        {"mean_PRI": 1550, "PRI_variance": 130, "PRI_pattern": "staggered",
         "mean_frequency": 3800, "frequency_variance": 68, "mean_pulse_width": 5.1,
         "duty_cycle": 0.0029, "radar_name": "search_radar"},
        
        {"mean_PRI": 1520, "PRI_variance": 125, "PRI_pattern": "jittered",
         "mean_frequency": 4900, "frequency_variance": 72, "mean_pulse_width": 4.7,
         "duty_cycle": 0.0030, "radar_name": "maritime_radar"},
        
        # Between fire_control and tracking (overlapping X-band)
        {"mean_PRI": 480, "PRI_variance": 18, "PRI_pattern": "constant",
         "mean_frequency": 8800, "frequency_variance": 28, "mean_pulse_width": 1.1,
         "duty_cycle": 0.0024, "radar_name": "fire_control_radar"},
        
        {"mean_PRI": 520, "PRI_variance": 15, "PRI_pattern": "constant",
         "mean_frequency": 8600, "frequency_variance": 24, "mean_pulse_width": 1.3,
         "duty_cycle": 0.0023, "radar_name": "tracking_radar"},
        
        {"mean_PRI": 880, "PRI_variance": 65, "PRI_pattern": "constant",
         "mean_frequency": 8200, "frequency_variance": 30, "mean_pulse_width": 2.2,
         "duty_cycle": 0.0025, "radar_name": "tracking_radar"},
        
        {"mean_PRI": 920, "PRI_variance": 70, "PRI_pattern": "constant",
         "mean_frequency": 1350, "frequency_variance": 42, "mean_pulse_width": 2.8,
         "duty_cycle": 0.0029, "radar_name": "air_traffic_control"},
        
        {"mean_PRI": 3200, "PRI_variance": 210, "PRI_pattern": "staggered",
         "mean_frequency": 2950, "frequency_variance": 95, "mean_pulse_width": 7.2,
         "duty_cycle": 0.0020, "radar_name": "weather_radar"},
        
        {"mean_PRI": 2800, "PRI_variance": 190, "PRI_pattern": "jittered",
         "mean_frequency": 3100, "frequency_variance": 88, "mean_pulse_width": 6.3,
         "duty_cycle": 0.0026, "radar_name": "search_radar"},
        
        {"mean_PRI": 3600, "PRI_variance": 180, "PRI_pattern": "staggered",
         "mean_frequency": 5100, "frequency_variance": 85, "mean_pulse_width": 6.5,
         "duty_cycle": 0.0019, "radar_name": "weather_radar"},
        
        {"mean_PRI": 1650, "PRI_variance": 135, "PRI_pattern": "jittered",
         "mean_frequency": 5200, "frequency_variance": 78, "mean_pulse_width": 4.6,
         "duty_cycle": 0.0030, "radar_name": "maritime_radar"},
    ]
    
    for _ in range(5):
        for case in edge_cases:
            noisy_case = case.copy()
            noisy_case["mean_PRI"] += random.randint(-100, 100)
            noisy_case["PRI_variance"] += random.randint(-15, 15)
            noisy_case["mean_frequency"] += random.randint(-200, 200)
            noisy_case["frequency_variance"] += random.randint(-10, 10)
            noisy_case["mean_pulse_width"] += round(random.uniform(-0.3, 0.3), 2)
            noisy_case["duty_cycle"] += round(random.uniform(-0.0005, 0.0005), 4)
            
            # Keep in bounds
            noisy_case["mean_PRI"] = max(100, noisy_case["mean_PRI"])
            noisy_case["PRI_variance"] = max(5, noisy_case["PRI_variance"])
            noisy_case["mean_frequency"] = max(1000, noisy_case["mean_frequency"])
            noisy_case["frequency_variance"] = max(10, noisy_case["frequency_variance"])
            noisy_case["mean_pulse_width"] = max(0.3, noisy_case["mean_pulse_width"])
            noisy_case["duty_cycle"] = max(0.0015, min(0.0035, noisy_case["duty_cycle"]))
            
            dataset.append(noisy_case)
    
    return dataset

dataset = []

dataset += generate_samples(150, "search_radar",
                            (1500, 2200), (80, 220), 
                            (2700, 3600), (40, 90),  
                            (4.5, 6.5), (0.0025, 0.0032),
                            ["staggered", "jittered"],
                            noise_level=0.3)  

dataset += generate_samples(150, "fire_control_radar",
                            (200, 500), (10, 40),    
                            (8500, 11000), (20, 50),
                            (0.5, 1.2), (0.0022, 0.0028),
                            ["constant"],
                            noise_level=0.3)

dataset += generate_samples(150, "tracking_radar",
                            (500, 900), (5, 25),     
                            (7500, 9500), (12, 30),  
                            (1.0, 2.0), (0.0019, 0.0024),
                            ["constant"],
                            noise_level=0.3)

dataset += generate_samples(150, "weather_radar",
                            (3800, 5200), (200, 380), 
                            (2700, 3300), (80, 130),  
                            (7.0, 10.0), (0.0016, 0.0020),
                            ["staggered", "jittered"],
                            noise_level=0.3)

dataset += generate_samples(150, "air_traffic_control",
                            (850, 1150), (60, 100),  
                            (1000, 1450), (30, 50), 
                            (2.5, 3.5), (0.0027, 0.0032),
                            ["constant"],
                            noise_level=0.3)

dataset += generate_samples(150, "maritime_radar",
                            (1300, 1700), (100, 140), 
                            (4500, 5400), (60, 80),  
                            (4.0, 5.0), (0.0028, 0.0032),
                            ["staggered", "jittered"],
                            noise_level=0.3)

dataset = add_edge_cases(dataset)

df = pd.DataFrame(dataset)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("data/radar_dataset_with_noise.csv", index=False)

print(f"Dataset generated: {df.shape}")
print(f"Total samples: {len(df)}")
print(f"\nClass distribution:")
print(df['radar_name'].value_counts())
