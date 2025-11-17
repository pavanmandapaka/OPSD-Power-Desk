import pandas as pd
import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
COUNTRIES = ['DE', 'FR', 'ES']

WINDOW_SIZE = 336
MIN_PERIODS = 168
Z_THRESHOLD = 3.0

def detect_anomalies():
    for country in COUNTRIES:
        print(f"Anomaly for {country}")
        input_path = os.path.join(OUTPUT_DIR, f"{country}_forecasts_test.csv")

        try:
            df = pd.read_csv(input_path, parse_dates=['timestamp'] )
        except FileNotFoundError:
            print(f"file not found for {country}, skipping...")
            continue

        df['error'] = df['y_true'] - df['yhat']

        rolling_mean = df['error'].rolling(window=WINDOW_SIZE , min_periods=MIN_PERIODS).mean()
        rolling_std = df['error'].rolling(window=WINDOW_SIZE, min_periods=MIN_PERIODS).std()

        df['z_score'] = (df['error'] - rolling_mean)/(rolling_std + 1e-6)

        df['flag_z'] = (df['z_score'].abs() >= Z_THRESHOLD).astype(int)

        output_cols = ['timestamp', 'y_true', 'yhat', 'error', 'z_score', 'flag_z']
        df_out = df[output_cols]
        output_path = os.path.join(OUTPUT_DIR, f"{country}_anomalies.csv")
        df_out.to_csv(output_path, index=False)

        n_anomalies = df['flag_z'].sum()
        print(f"Detected {n_anomalies} anomalies for {country}. Saved to {output_path}")

if __name__ == "__main__":
    detect_anomalies()