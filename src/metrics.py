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

