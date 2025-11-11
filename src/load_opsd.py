import pandas as pd
import os

# Define the project's root directory
# This makes the script runnable from anywhere
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'time_series_60min_singleindex.csv') # Use your exact raw file name
SAVE_DIR = os.path.join(PROJECT_ROOT, 'data')

# 1. Choose your 3 countries [cite: 2, 14]
COUNTRIES = ['DE', 'FR', 'ES'] # Example: Germany, France, Spain

# 2. Define the columns to extract and their new names [cite: 15-22]
# We want load, plus optional wind and solar
COLUMN_MAP = {
    'utc_timestamp': 'timestamp',
    'DE_load_actual_entsoe_transparency': 'DE_load',
    'DE_wind_generation_actual': 'DE_wind',
    'DE_solar_generation_actual': 'DE_solar',

    'FR_load_actual_entsoe_transparency': 'FR_load',
    'FR_wind_onshore_generation_actual': 'FR_wind',
    'FR_solar_generation_actual': 'FR_solar',

    'ES_load_actual_entsoe_transparency': 'ES_load',
    'ES_wind_onshore_generation_actual': 'ES_wind',
    'ES_solar_generation_actual': 'ES_solar',
}

# 3. Read the raw data
print(f"Loading raw data from {RAW_DATA_PATH}...")
# We only read the columns we care about to save memory
try:
    df_raw = pd.read_csv(RAW_DATA_PATH, usecols=COLUMN_MAP.keys())
except FileNotFoundError:
    print(f"Error: Raw data file not found at {RAW_DATA_PATH}")
    print("Please download the OPSD hourly data and place it in the data/ folder.")
    exit()

# Rename columns to simpler names from our map
df_raw = df_raw.rename(columns=COLUMN_MAP)

# 4. Process and save one file per country
for country in COUNTRIES:
    print(f"Processing data for {country}...")

    # Select columns for this country (e.g., timestamp, DE_load, DE_wind, DE_solar)
    country_cols = ['timestamp'] + [col for col in df_raw.columns if col.startswith(country)]

    # Create a new, tidy DataFrame for this country [cite: 24]
    df_country = df_raw[country_cols].copy()

    # Rename columns to the standard 'load', 'wind', 'solar' [cite: 18, 25]
    rename_dict = {
        f'{country}_load': 'load',
        f'{country}_wind': 'wind',
        f'{country}_solar': 'solar'
    }
    df_country = df_country.rename(columns=rename_dict)

    # 5. Clean the data as required

    # Drop rows where the 'load' is missing 
    df_country = df_country.dropna(subset=['load'])

    # Convert timestamp to datetime object (crucial for time series)
    df_country['timestamp'] = pd.to_datetime(df_country['timestamp'])

    # Sort by timestamp to ensure chronological order 
    df_country = df_country.sort_values(by='timestamp')

    # 6. Save the clean data 
    save_path = os.path.join(SAVE_DIR, f'{country}.csv')
    df_country.to_csv(save_path, index=False)
    print(f"Saved clean data for {country} to {save_path}")

print("Day 1 data processing complete.")