import pandas as pd
import statsmodels.api as sm
import os
import warnings
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

os.makedirs(OUTPUT_DIR, exist_ok=True)
HORIZON = 24 
STRIDE = 24 * 7

COUNTRIES = ['DE', 'FR', 'ES']

BEST_ORDERS = {
    'DE': {'order': (2, 1, 2), 'seasonal_order': (1, 1, 1, 24)},
    'FR': {'order': (1, 1, 2), 'seasonal_order': (1, 1, 1, 24)},
    'ES': {'order': (2, 1, 1), 'seasonal_order': (1, 1, 1, 24)},
}

EXOG_VARS = ['wind', 'solar']
print("Forecasting script started.")

def run_backtest():
    warnings.filterwarnings("ignore")

    for country in COUNTRIES:
        print(f"--- Processing {country} ---")

        file_path = os.path.join(DATA_DIR, f"{country}.csv")
        try:
            df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
        except FileNotFoundError:
            print(f"Data file not found for {country}. Skipping.")
            continue

        endog = df['load']
        exog = df[EXOG_VARS] if EXOG_VARS else None

        n = len(endog)
        train_end_idx = int(n * 0.8)
        dev_end_idx = int(n * 0.9)

        splits_to_run = {
            'dev': (train_end_idx, dev_end_idx),
            'test': (dev_end_idx, n)
        }

        try:
            orders = BEST_ORDERS[country]
        except KeyError:
            print(f"No SARIMA orders defined for {country}. Skipping.")
            continue

        for split_name, (start_idx, end_idx) in splits_to_run.items():
            print(f"Running backtest on {split_name} set...")

            all_forecasts = []

            for i in tqdm(range(start_idx, end_idx, STRIDE)):

                lookback = 30 * 24
                train_start = max(0, i - lookback)
                current_train_endog = endog.iloc[train_start:i]

                if len(current_train_endog) < 7*24:
                    continue

                current_train_exog = exog.iloc[train_start:i] if exog is not None else None

                try:
                    model = sm.tsa.SARIMAX(
                        current_train_endog,
                        exog=current_train_exog,
                        order=orders['order'],
                        seasonal_order=orders['seasonal_order'],
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    res = model.fit(disp=False, maxiter=50, method='lbfgs')

                    future_exog = exog.iloc[i:i+HORIZON] if exog is not None else None

                    forecast = res.get_forecast(steps=HORIZON, exog=future_exog)

                    yhat = forecast.predicted_mean

                    conf_int = forecast.conf_int(alpha=0.20)

                    forecast_df = pd.DataFrame({
                        'timestamp': yhat.index,
                        'yhat': yhat.values,
                        'lo': conf_int.iloc[:, 0].values,
                        'hi': conf_int.iloc[:, 1].values,
                        'horizon': range(1, HORIZON + 1),
                        'train_end': current_train_endog.index[-1]
                    })
                    all_forecasts.append(forecast_df)

                except Exception as e:
                    print(f"Error at timestamp {endog.index[i]} for {country}: {e}")
                    continue

            if not all_forecasts:
                print(f"No forecasts generated for {country} {split_name} set.")
                continue

            final_forecast_df = pd.concat(all_forecasts, ignore_index=True)

            final_forecast_df = final_forecast_df.set_index('timestamp').join(endog.rename('y_true')).reset_index()

            final_cols = ['timestamp', 'y_true', 'yhat', 'lo', 'hi', 'horizon', 'train_end']
            final_forecast_df = final_forecast_df[final_cols]

            save_path = os.path.join(OUTPUT_DIR, f"{country}_forecasts_{split_name}.csv")
            final_forecast_df.to_csv(save_path, index=False)
            print(f"Saved {split_name} forecasts for {country} to {save_path}")

if __name__ == "__main__":
    run_backtest()
    print("Forecasting script finished.")