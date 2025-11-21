import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

st.set_page_config(
    page_title="OPSD PowerDesk - Live Monitoring",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')


@st.cache_data
def load_data(country):
    """Loads the live simulation outputs for the selected country."""
    
    forecast_path = os.path.join(OUTPUT_DIR, f'{country}_live_forecasts.csv')
    if not os.path.exists(forecast_path):
        return None, None
        
    df_fc = pd.read_csv(forecast_path, parse_dates=['timestamp'])
    df_fc = df_fc.sort_values('timestamp')
    
    log_path = os.path.join(OUTPUT_DIR, f'{country}_online_updates.csv')
    if os.path.exists(log_path):
        df_log = pd.read_csv(log_path, parse_dates=['timestamp'])
        df_log = df_log.sort_values('timestamp', ascending=False)
    else:
        df_log = pd.DataFrame()
        
    return df_fc, df_log


def calculate_rolling_mase(df, window_hours=168):
    """Calculate rolling 7-day MASE."""
    if len(df) < window_hours:
        window_hours = len(df)
    
    last_window = df.iloc[-window_hours:]
    
    # MAE of forecast
    mae_fc = (last_window['y_true'] - last_window['yhat']).abs().mean()
    
    # MAE of naive forecast (24h lag)
    naive_errors = (last_window['y_true'] - last_window['y_true'].shift(24)).abs()
    mae_naive = naive_errors.mean()
    
    mase = mae_fc / (mae_naive + 1e-6)
    return mase


def calculate_pi_coverage(df, window_hours=168):
    """Calculate 80% prediction interval coverage for last 7 days."""
    if len(df) < window_hours:
        window_hours = len(df)
    
    last_window = df.iloc[-window_hours:]
    
    # Check if confidence intervals exist
    if 'yhat_lower' in df.columns and 'yhat_upper' in df.columns:
        in_interval = (last_window['y_true'] >= last_window['yhat_lower']) & \
                      (last_window['y_true'] <= last_window['yhat_upper'])
        coverage = (in_interval.sum() / len(last_window)) * 100
    else:
        # Estimate 80% PI using historical errors
        residuals = df['y_true'] - df['yhat']
        std = residuals.std()
        z_80 = 1.28  # Z-score for 80% PI
        
        yhat_lower_est = last_window['yhat'] - z_80 * std
        yhat_upper_est = last_window['yhat'] + z_80 * std
        
        in_interval = (last_window['y_true'] >= yhat_lower_est) & \
                      (last_window['y_true'] <= yhat_upper_est)
        coverage = (in_interval.sum() / len(last_window)) * 100
    
    return coverage


def count_anomalies_today(df):
    """Count anomalies in the last 24 hours."""
    if len(df) < 24:
        return 0
    
    last_24h = df.iloc[-24:]
    
    # Check for flag_z column
    if 'flag_z' in df.columns:
        anomalies = last_24h['flag_z'].sum()
        
        # Also check flag_cusum if present
        if 'flag_cusum' in df.columns:
            cusum_anomalies = last_24h['flag_cusum'].sum()
            return int(anomalies + cusum_anomalies)
        
        return int(anomalies)
    else:
        # Fallback: calculate Z-score based anomalies
        residuals = df['y_true'] - df['yhat']
        rolling_std = residuals.rolling(336, min_periods=24).std()
        
        last_24h_residuals = residuals.iloc[-24:]
        last_std = rolling_std.iloc[-24:].mean()
        
        anomalies = (last_24h_residuals.abs() > 3 * last_std).sum()
        return int(anomalies)


def get_anomaly_points(df, lookback_hours=336):
    """Get anomaly points for visualization."""
    if len(df) < lookback_hours:
        lookback_hours = len(df)
    
    subset = df.iloc[-lookback_hours:]
    
    # Use flag_z if available
    if 'flag_z' in df.columns:
        anomalies = subset[subset['flag_z'] == 1].copy()
    else:
        # Calculate on the fly
        residuals = subset['y_true'] - subset['yhat']
        rolling_std = residuals.rolling(168, min_periods=24).std()
        threshold = 3 * rolling_std
        
        is_anomaly = residuals.abs() > threshold
        anomalies = subset[is_anomaly].copy()
    
    return anomalies


def create_forecast_cone(df, hours_ahead=24):
    """Create forecast cone for next 24 hours."""
    last_timestamp = df['timestamp'].iloc[-1]
    last_yhat = df['yhat'].iloc[-1]
    
    # Generate future timestamps
    future_times = pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1),
        periods=hours_ahead,
        freq='h'
    )
    
    # Estimate uncertainty based on historical errors
    residuals = df['y_true'] - df['yhat']
    std_error = residuals.std()
    
    # For 80% PI: z = 1.28
    z_80 = 1.28
    
    # Project forward with increasing uncertainty
    mean_forecast = np.full(hours_ahead, last_yhat)
    
    # Gradually increase uncertainty
    uncertainty = std_error * z_80 * np.linspace(1, 1.5, hours_ahead)
    
    upper_bound = mean_forecast + uncertainty
    lower_bound = mean_forecast - uncertainty
    
    return future_times, mean_forecast, upper_bound, lower_bound


st.title("OPSD PowerDesk - Live Load Forecasting")
st.markdown("Real-time monitoring of electricity load predictions with SARIMA models")

st.sidebar.title("Country Selection")
country = st.sidebar.selectbox(
    "Select Country",
    ["DE", "FR", "ES"],
    index=0,  # Preselect Germany (live country)
    help="Choose a country to view live forecasts"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Dashboard Info")
st.sidebar.info(
    "This dashboard displays:\n"
    "- Last 7-14 days of actual vs forecast\n"
    "- Next 24h forecast with 80% PI\n"
    "- Anomaly detection highlights\n"
    "- Rolling performance metrics"
)

# Load Data
df, logs = load_data(country)

if df is None:
    st.error(f"No data found for **{country}**")
    st.info("Run the live simulation first: `python3 src/run_all_simulations.py`")
    st.stop()

st.markdown("---")
st.subheader(f"Key Performance Indicators - {country}")

# Calculate metrics
rolling_mase = calculate_rolling_mase(df, window_hours=168)
pi_coverage = calculate_pi_coverage(df, window_hours=168)
anomalies_today = count_anomalies_today(df)

# Get last update info
if not logs.empty:
    last_update = logs.iloc[0]
    last_update_time = last_update['timestamp']
    last_update_reason = last_update['event']
    last_update_display = last_update_time.strftime('%Y-%m-%d %H:%M')
else:
    last_update_time = "N/A"
    last_update_reason = "No updates logged"
    last_update_display = "N/A"

# Display KPI tiles
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Rolling 7D MASE",
        value=f"{rolling_mase:.3f}",
        delta=f"{'Good' if rolling_mase < 1.0 else 'Check'}",
        delta_color="inverse"
    )
    st.caption("Lower is better (< 1.0 beats naive)")

with col2:
    st.metric(
        label="80% PI Coverage (7D)",
        value=f"{pi_coverage:.1f}%",
        delta=f"{'Good' if pi_coverage >= 75 else 'Low'}",
        delta_color="normal" if pi_coverage >= 75 else "inverse"
    )
    st.caption("Target: 75-85%")

with col3:
    st.metric(
        label="Anomalies (Last 24h)",
        value=f"{anomalies_today}",
        delta=f"{'Normal' if anomalies_today < 5 else 'High'}",
        delta_color="inverse"
    )
    st.caption("High Z-score errors")

with col4:
    st.metric(
        label="Last Update",
        value=last_update_display,
        delta=last_update_reason,
        delta_color="off"
    )
    st.caption("Model update status")

# 4. Main Chart: Live Series & Forecast Cone
st.markdown("---")
st.subheader(f"Live Load Monitoring - {country}")

# Time window selector
col_left, col_right = st.columns([3, 1])
with col_left:
    window_days = st.slider(
        "History Window",
        min_value=7,
        max_value=30,
        value=14,
        step=1,
        help="Number of days to display"
    )
    window_hours = window_days * 24

with col_right:
    show_anomalies = st.checkbox("Show Anomalies", value=True)
    show_forecast_cone = st.checkbox("Show 24h Forecast Cone", value=True)

# Get data subset
if len(df) > window_hours:
    subset = df.iloc[-window_hours:].copy()
else:
    subset = df.copy()

# Create figure
fig = go.Figure()

# 1. Actual Load (y_true)
fig.add_trace(go.Scatter(
    x=subset['timestamp'],
    y=subset['y_true'],
    name='Actual Load',
    line=dict(color='#1f77b4', width=2.5),
    mode='lines'
))

# 2. Forecast (yhat)
fig.add_trace(go.Scatter(
    x=subset['timestamp'],
    y=subset['yhat'],
    name='SARIMA Forecast',
    line=dict(color='#ff7f0e', width=2, dash='dot'),
    mode='lines'
))

# 3. Anomaly Tape (highlight anomalous hours)
if show_anomalies:
    anomaly_points = get_anomaly_points(df, lookback_hours=window_hours)
    
    if len(anomaly_points) > 0:
        fig.add_trace(go.Scatter(
            x=anomaly_points['timestamp'],
            y=anomaly_points['y_true'],
            name='Anomaly (flag_z=1)',
            mode='markers',
            marker=dict(
                color='red',
                size=10,
                symbol='x',
                line=dict(width=2, color='darkred')
            )
        ))

# 4. Forecast Cone (Next 24h with 80% PI)
if show_forecast_cone:
    future_times, mean_fc, upper, lower = create_forecast_cone(df, hours_ahead=24)
    
    # Upper bound
    fig.add_trace(go.Scatter(
        x=future_times,
        y=upper,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Lower bound (fill between)
    fig.add_trace(go.Scatter(
        x=future_times,
        y=lower,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 127, 14, 0.2)',
        name='Next 24h (80% PI)',
        hoverinfo='skip'
    ))
    
    # Mean forecast line
    fig.add_trace(go.Scatter(
        x=future_times,
        y=mean_fc,
        name='24h Ahead Forecast',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        mode='lines'
    ))

# Layout
fig.update_layout(
    height=600,
    xaxis_title="Timestamp (UTC)",
    yaxis_title="Load (MW)",
    hovermode='x unified',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    template="plotly_white"
)

st.plotly_chart(fig, width='stretch')

# 5. Update Status & System Log
st.markdown("---")
st.subheader("System Activity Log")

if not logs.empty:
    # Show recent updates
    st.markdown(f"**Latest Update:** {last_update_display}")
    st.markdown(f"**Reason:** `{last_update_reason}`")
    
    st.markdown("##### Recent Updates")
    
    # Format logs for display
    display_logs = logs.head(20).copy()
    display_logs['timestamp'] = display_logs['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    st.dataframe(
        display_logs,
        width='stretch',
        hide_index=True
    )
else:
    st.info("No update logs available yet.")