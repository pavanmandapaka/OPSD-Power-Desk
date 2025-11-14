import pandas as pd
import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'time_series_60min_singleindex.csv') # Use your exact raw file name
SAVE_DIR = os.path.join(PROJECT_ROOT, 'data')

COUNTRIES = ['DE', 'FR', 'ES']

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

print(f"Loading raw data from {RAW_DATA_PATH}...")
try:
    df_raw = pd.read_csv(RAW_DATA_PATH, usecols=COLUMN_MAP.keys())
except FileNotFoundError:
    print(f"Error: Raw data file not found at {RAW_DATA_PATH}")
    print("Please download the OPSD hourly data and place it in the data/ folder.")
    exit()

df_raw = df_raw.rename(columns=COLUMN_MAP)

for country in COUNTRIES:
    print(f"Processing data for {country}...")

    country_cols = ['timestamp'] + [col for col in df_raw.columns if col.startswith(country)]

    df_country = df_raw[country_cols].copy()

    rename_dict = {
        f'{country}_load': 'load',
        f'{country}_wind': 'wind',
        f'{country}_solar': 'solar'
    }
    df_country = df_country.rename(columns=rename_dict)


    df_country = df_country.dropna(subset=['load'])

    df_country['timestamp'] = pd.to_datetime(df_country['timestamp'])

    df_country = df_country.sort_values(by='timestamp')

    save_path = os.path.join(SAVE_DIR, f'{country}.csv')
    df_country.to_csv(save_path, index=False)
    print(f"Saved clean data for {country} to {save_path}")

print("Day 1 data processing complete.")