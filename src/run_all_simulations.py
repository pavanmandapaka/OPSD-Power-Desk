import pandas as pd
import numpy as np
import os
import warnings
from tqdm import tqdm
import statsmodels.api as sm 

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

SIM_START_HOURS = 2000
HISTORY_WINDOW = 336
DRIFT_WINDOW = 720
REFIT_WINDOW = 2160 

COUNTRY_CONFIGS = {
    'DE': {
        'order': (2, 1, 2),
        'seasonal_order': (1, 1, 1, 24)
    },
    'FR': {
        'order': (1, 1, 1),
        'seasonal_order': (1, 1, 1, 24)
    },
    'ES': {
        'order': (1, 1, 1),
        'seasonal_order': (1, 1, 1, 24)
    }
}

class LiveSimulator:
    def __init__(self, country_code, model_order, seasonal_order):
        self.country = country_code
        self.model_order = model_order
        self.seasonal_order = seasonal_order
        
        self.current_step = 0
        self.z_scores_history = []
        self.ewma_z = 0.0
        self.alpha = 0.1

        self.logs = []
        self.forecasts = [] 

        self.model_res = None 
        
        self.load_data()

    def load_data(self):
        print(f"Loading data for {self.country}...")
        data_path = os.path.join(DATA_DIR, f'{self.country}.csv')
        self.full_df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
        n = len(self.full_df)
        test_start_idx = int(n * 0.9)
        self.history = self.full_df.iloc[:test_start_idx].copy()
        self.future_data = self.full_df.iloc[test_start_idx:]

    def refit_model(self):
        if len(self.history) > REFIT_WINDOW:
            train_data = self.history['load'].iloc[-REFIT_WINDOW:]
        else:
            train_data = self.history['load'] 

        model = sm.tsa.SARIMAX(
            train_data,
            order=self.model_order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        try:
            self.model_res = model.fit(disp=False, maxiter=50)
        except Exception as e:
            print(f"Refit failed: {e}")

    def make_forecast(self, current_time):
        if self.model_res is None:
            self.refit_model()

        try:
            forecast_obj = self.model_res.get_forecast(steps=24)
            yhat = forecast_obj.predicted_mean
            conf_int = forecast_obj.conf_int(alpha=0.2)

            forecast_idx = pd.date_range(start=current_time + pd.Timedelta(hours=1), periods=24, freq='h')

            yhat.index = forecast_idx
            conf_int.index = forecast_idx

            return pd.DataFrame({
                'yhat': yhat,
                'lo': conf_int.iloc[:, 0],
                'hi': conf_int.iloc[:, 1]
            })

        except Exception as e:
            print(f"Forecast failed at {current_time}: {e}")
            last_val = self.history['load'].iloc[-1]
            forecast_idx = pd.date_range(start=current_time + pd.Timedelta(hours=1), periods=24, freq='h')
            return pd.DataFrame({'yhat': [last_val]*24}, index=forecast_idx)

    def run_simulation(self):
        print(f"Starting simulation for {self.country} with order={self.model_order}, seasonal_order={self.seasonal_order}")
        print(f"Running for {min(len(self.future_data), SIM_START_HOURS)} steps...")

        self.refit_model()
        current_forecast_df = self.make_forecast(self.history.index[-1])

        steps_to_run = min(len(self.future_data), SIM_START_HOURS)

        for i in tqdm(range(steps_to_run), desc=f"{self.country}"):
            new_row = self.future_data.iloc[[i]]
            current_time = new_row.index[0]
            actual_load = new_row['load'].values[0]
            self.history = pd.concat([self.history, new_row])

            try:
                pred_load = current_forecast_df.loc[current_time, 'yhat']
                lo_bound = current_forecast_df.loc[current_time, 'lo']
                hi_bound = current_forecast_df.loc[current_time, 'hi']

                self.forecasts.append({
                    'timestamp': current_time,
                    'y_true': actual_load,
                    'yhat': pred_load,
                    'yhat_lo': lo_bound,
                    'yhat_hi': hi_bound
                })

                residual = actual_load - pred_load

                recent_hist = self.history['load'].iloc[-HISTORY_WINDOW:]
                diffs = (recent_hist - recent_hist.shift(24)).dropna()
                mu = diffs.mean()
                sigma = diffs.std() + 1e-6

                z_score = (residual - mu) / sigma
                self.z_scores_history.append(abs(z_score))

            except KeyError:
                z_score = 0

            is_drift, thresh = self.check_drift(z_score)

            update_type = None

            if is_drift:
                update_type = 'DRIFT_ADAPTATION'
                self.refit_model()
                current_forecast_df = self.make_forecast(current_time)

            elif current_time.hour == 0:
                update_type = 'SCHEDULED_REFIT'
                self.refit_model()
                current_forecast_df = self.make_forecast(current_time)

            if update_type:
                self.logs.append({
                    'timestamp': current_time,
                    'event': update_type,
                    'details': f'EWMA: {self.ewma_z:.2f} | Thresh: {thresh:.2f}'
                })

        logs_df = pd.DataFrame(self.logs)
        logs_df.to_csv(os.path.join(OUTPUT_DIR, f'{self.country}_online_updates.csv'), index=False)

        forecasts_df = pd.DataFrame(self.forecasts)
        forecasts_df.to_csv(os.path.join(OUTPUT_DIR, f'{self.country}_live_forecasts.csv'), index=False)

        print(f"\n{self.country} Simulation Complete.")
        print(f"Total Updates: {len(logs_df)}")
        if len(logs_df) > 0:
            print(logs_df['event'].value_counts())
        print("-" * 50)

    def check_drift(self, current_z):
        abs_z = abs(current_z)
        self.ewma_z = (self.alpha * abs_z) + ((1 - self.alpha) * self.ewma_z)
        if len(self.z_scores_history) < DRIFT_WINDOW:
            threshold = 3.0
        else:
            threshold = np.percentile(self.z_scores_history[-DRIFT_WINDOW:], 95)
        return (self.ewma_z > threshold), threshold


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        countries_to_run = sys.argv[1].split(',')
    else:
        countries_to_run = list(COUNTRY_CONFIGS.keys())
    
    print("=" * 50)
    print(f"Running Live Simulations for: {', '.join(countries_to_run)}")
    print("=" * 50)
    
    for country in countries_to_run:
        if country not in COUNTRY_CONFIGS:
            print(f"Warning: {country} not found in configurations, skipping...")
            continue
            
        config = COUNTRY_CONFIGS[country]
        print(f"\n{'='*50}")
        print(f"Processing: {country}")
        print(f"{'='*50}")
        
        sim = LiveSimulator(
            country_code=country,
            model_order=config['order'],
            seasonal_order=config['seasonal_order']
        )
        print(f"History: {len(sim.history)}h, Future: {len(sim.future_data)}h")
        sim.run_simulation()
    
    print("\n" + "="*50)
    print("All simulations complete!")
    print("="*50)
