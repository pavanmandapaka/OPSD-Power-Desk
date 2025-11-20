import pandas as pd
import numpy as np
import os
import warnings
from tqdm import tqdm
import statsmodels.api as sm # <--- NEW

# --- Config ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

COUNTRY = 'DE'
SIM_START_HOURS = 2000
HISTORY_WINDOW = 336
DRIFT_WINDOW = 720
REFIT_WINDOW = 2160 # 90 days * 24h = 2160h (Training data size for refit)

# Copy the BEST orders you found on Day 3!
# Example for DE:
MODEL_ORDER = (2, 1, 2) 
SEASONAL_ORDER = (1, 1, 1, 24)

class LiveSimulator:
    def __init__(self, country_code):
        self.country = country_code
        
        # Simulation State
        self.current_step = 0
        self.z_scores_history = []
        self.ewma_z = 0.0
        self.alpha = 0.1

        self.logs = []
        self.forecasts = [] # <--- NEW: To store all forecasts for analysis

        # The Current Model
        self.model_res = None # We will store the fitted model result here
        
        self.load_data()

    def load_data(self):
        # (Keep this the same as Day 10)
        print(f"Loading data for {self.country}...")
        data_path = os.path.join(DATA_DIR, f'{self.country}.csv')
        self.full_df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
        n = len(self.full_df)
        test_start_idx = int(n * 0.9)
        self.history = self.full_df.iloc[:test_start_idx].copy()
        self.future_data = self.full_df.iloc[test_start_idx:]

    def refit_model(self):
        """
        Retrains the SARIMA model on the last 90 days (REFIT_WINDOW) of history.
        """
        # 1. Select Training Data (Rolling Window)
        if len(self.history) > REFIT_WINDOW:
            train_data = self.history['load'].iloc[-REFIT_WINDOW:]
        else:
            train_data = self.history['load'] # Use all if < 90 days

        # 2. Define Model
        # We use the fixed orders we found on Day 3
        model = sm.tsa.SARIMAX(
            train_data,
            order=MODEL_ORDER,
            seasonal_order=SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        # 3. Fit Model
        # disp=False prevents console spam
        try:
            self.model_res = model.fit(disp=False)
        except Exception as e:
            print(f"Refit failed: {e}")
            # If fail, keep old model or do nothing

    def make_forecast(self, current_time):
        """
        Uses the currently fitted model to predict the next 24 hours.
        """
        # If we haven't trained yet (start of sim), train now
        if self.model_res is None:
            self.refit_model()

        # Forecast next 24 steps
        try:
            # The model knows where it stopped. We forecast from end of training.
            # Note: In a pure loop, we might need to append new observations 
            # to the model object without full retraining, but statsmodels 
            # .append() is tricky. For simplicity in this project, 
            # we will rely on the daily refit or use the 'apply' method 
            # if available, but strictly speaking, we just need the forecast here.

            # Simplified approach: We forecast from the model's last state.
            # Since we refit daily, the model is reasonably fresh.
            forecast_obj = self.model_res.get_forecast(steps=24)
            yhat = forecast_obj.predicted_mean
            conf_int = forecast_obj.conf_int(alpha=0.2)

            # Create DataFrame
            forecast_idx = pd.date_range(start=current_time + pd.Timedelta(hours=1), periods=24, freq='h')

            # Align index just in case
            yhat.index = forecast_idx
            conf_int.index = forecast_idx

            return pd.DataFrame({
                'yhat': yhat,
                'lo': conf_int.iloc[:, 0],
                'hi': conf_int.iloc[:, 1]
            })

        except Exception as e:
            print(f"Forecast failed at {current_time}: {e}")
            # Fallback: Persist last known value (Naive)
            last_val = self.history['load'].iloc[-1]
            forecast_idx = pd.date_range(start=current_time + pd.Timedelta(hours=1), periods=24, freq='h')
            return pd.DataFrame({'yhat': [last_val]*24}, index=forecast_idx)

    def run_simulation(self):
        print(f"Starting Smart Simulation for {min(len(self.future_data), SIM_START_HOURS)} steps...")

        # Initial Training
        self.refit_model()
        current_forecast_df = self.make_forecast(self.history.index[-1])

        steps_to_run = min(len(self.future_data), SIM_START_HOURS)

        for i in tqdm(range(steps_to_run)):
            # 1. INGEST
            new_row = self.future_data.iloc[[i]]
            current_time = new_row.index[0]
            actual_load = new_row['load'].values[0]
            self.history = pd.concat([self.history, new_row])

            # 2. COMPUTE STATS
            try:
                # Get prediction for THIS hour from our active forecast
                pred_load = current_forecast_df.loc[current_time, 'yhat']

                # Save the prediction for later analysis
                self.forecasts.append({
                    'timestamp': current_time,
                    'y_true': actual_load,
                    'yhat': pred_load
                })

                residual = actual_load - pred_load

                # Calculate Z-Score (using last 14 days history)
                recent_hist = self.history['load'].iloc[-HISTORY_WINDOW:]
                # Seasonal differencing for Z-score calc implies comparing to 24h ago
                # Simplified: just use standard deviation of raw load or residuals if stored
                # Better: Keep a running list of residuals like in Day 8
                # For speed here: we use a simple moving std of the *load* # minus the load 24h ago (naive residual)
                diffs = (recent_hist - recent_hist.shift(24)).dropna()
                mu = diffs.mean()
                sigma = diffs.std() + 1e-6

                # We compare our *actual* residual to this distribution
                z_score = (residual - mu) / sigma
                self.z_scores_history.append(abs(z_score))

            except KeyError:
                z_score = 0

            # 3. CHECK DRIFT
            is_drift, thresh = self.check_drift(z_score)

            # 4. ADAPTATION LOGIC
            update_type = None

            # Priority 1: Drift Trigger (Emergency Refit)
            if is_drift:
                update_type = 'DRIFT_ADAPTATION'
                self.refit_model()
                # After refit, update the forecast immediately
                current_forecast_df = self.make_forecast(current_time)

            # Priority 2: Scheduled Daily Refit (at 00:00)
            elif current_time.hour == 0:
                update_type = 'SCHEDULED_REFIT'
                self.refit_model()
                current_forecast_df = self.make_forecast(current_time)

            # Log if an update happened
            if update_type:
                self.logs.append({
                    'timestamp': current_time,
                    'event': update_type,
                    'details': f'EWMA: {self.ewma_z:.2f} | Thresh: {thresh:.2f}'
                })

        # 5. SAVE OUTPUTS
        # Save Logs
        logs_df = pd.DataFrame(self.logs)
        logs_df.to_csv(os.path.join(OUTPUT_DIR, f'{self.country}_online_updates.csv'), index=False)

        # Save Forecast History (for Day 12 Dashboard)
        forecasts_df = pd.DataFrame(self.forecasts)
        forecasts_df.to_csv(os.path.join(OUTPUT_DIR, f'{self.country}_live_forecasts.csv'), index=False)

        print("Simulation Complete.")
        print(f"Total Updates: {len(logs_df)}")
        print(logs_df['event'].value_counts())

    def check_drift(self, current_z):
        # (Keep the same as Day 10)
        abs_z = abs(current_z)
        self.ewma_z = (self.alpha * abs_z) + ((1 - self.alpha) * self.ewma_z)
        if len(self.z_scores_history) < DRIFT_WINDOW:
            threshold = 3.0
        else:
            threshold = np.percentile(self.z_scores_history[-DRIFT_WINDOW:], 95)
        return (self.ewma_z > threshold), threshold


if __name__ == '__main__':
    print("Simulator class defined.")
    sim = LiveSimulator(COUNTRY)
    print(f"Ready to simulate. History: {len(sim.history)}h. Future: {len(sim.future_data)}h")
    sim.run_simulation()
