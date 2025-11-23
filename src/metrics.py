import numpy as np
import pandas as pd


def calculate_mse(y_true, y_pred):
    """(MSE)."""
    return np.mean((y_true - y_pred)**2)

def calculate_rmse(y_true, y_pred):
    """(RMSE)."""
    return np.sqrt(calculate_mse(y_true, y_pred))

def calculate_mape(y_true, y_pred):
    """(MAPE)."""
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100

def calculate_smape(y_true, y_pred):
    """(sMAPE)."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-6
    return np.mean(numerator / denominator) * 100

def calculate_coverage(y_true, lo, hi):
    """Calculates the 80% Prediction Interval (PI) Coverage."""
    return np.mean((y_true >= lo) & (y_true <= hi)) * 100


def calculate_mase(y_true, y_pred, train_series, seasonality=24):
    """
    Calculates the Mean Absolute Scaled Error (MASE).
    This is the PRIMARY metric for the project.
    """

    mae_forecast = np.mean(np.abs(y_true - y_pred))


    naive_forecast_error = train_series.diff(seasonality).dropna()
    mae_naive = np.mean(np.abs(naive_forecast_error))

    if mae_naive == 0:
        return np.inf 

    return mae_forecast / mae_naive


def update_sarima_coverage(output_dir=None):
    """
    Update metrics_summary.csv with 80% PI Coverage for SARIMA live forecasts.
    Reads forecast files with confidence intervals and calculates coverage.
    """
    import os
    
    if output_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, 'outputs')
    
    metrics_file = os.path.join(output_dir, 'metrics_summary.csv')
    df = pd.read_csv(metrics_file)
    
    print("Calculating coverage for SARIMA live forecasts...")
    print("="*70)
    
    countries = ['DE', 'FR', 'ES']
    for country in countries:
        try:
            live_file = os.path.join(output_dir, f'{country}_live_forecasts.csv')
            df_live = pd.read_csv(live_file, parse_dates=['timestamp'])
            
            if 'yhat_lo' not in df_live.columns or 'yhat_hi' not in df_live.columns:
                print(f"{country}: No confidence intervals found, skipping...")
                continue
            
            df_live = df_live.dropna(subset=['y_true', 'yhat_lo', 'yhat_hi'])
            
            coverage = calculate_coverage(
                df_live['y_true'].values,
                df_live['yhat_lo'].values,
                df_live['yhat_hi'].values
            )
            
            mask = (df['country'] == country) & (df['split'] == 'sarima_live')
            df.loc[mask, 'Coverage (80%)'] = coverage
            
            print(f"{country}: Coverage = {coverage:.2f}%")
        except Exception as e:
            print(f"{country}: Error - {str(e)}")
    
    df.to_csv(metrics_file, index=False)
    
    print("="*70)
    print(f" Updated metrics saved to: {metrics_file}")
    
    print("\nUpdated SARIMA metrics:")
    sarima_metrics = df[df['split'] == 'sarima_live'][['country', 'MASE', 'sMAPE', 'Coverage (80%)']]
    print(sarima_metrics.to_string(index=False))
    
    return df


if __name__ == '__main__':
    update_sarima_coverage()

