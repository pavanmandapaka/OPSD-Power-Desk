import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# --- Config ---
st.set_page_config(page_title="OPSD PowerDesk", layout="wide")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')


@st.cache_data
def load_data(country):
    """Loads the live simulation outputs for the selected country."""
    
    # 1. Load the Forecast History (From Day 11)
    forecast_path = os.path.join(OUTPUT_DIR, f'{country}_live_forecasts.csv')
    if not os.path.exists(forecast_path):
        return None, None
        
    df_fc = pd.read_csv(forecast_path, parse_dates=['timestamp'])
    df_fc = df_fc.sort_values('timestamp')
    
    # 2. Load the Update Logs (From Day 10/11)
    log_path = os.path.join(OUTPUT_DIR, f'{country}_online_updates.csv')
    if os.path.exists(log_path):
        df_log = pd.read_csv(log_path, parse_dates=['timestamp'])
        df_log = df_log.sort_values('timestamp', ascending=False) # Newest first
    else:
        df_log = pd.DataFrame()
        
    return df_fc, df_log

def calculate_kpis(df):
    """Calculates Rolling 7-Day Metrics."""
    # Filter to last 7 days (168 hours)
    last_7d = df.iloc[-168:]
    
    # MASE Calculation (Simplified for Dashboard)
    # MAE of Forecast
    mae_fc = (last_7d['y_true'] - last_7d['yhat']).abs().mean()
    # MAE of Naive (Load - Load_24h_ago) - approximated from the series itself
    naive_diff = (last_7d['y_true'] - last_7d['y_true'].shift(24)).abs().mean()
    
    mase = mae_fc / (naive_diff + 1e-6)
    
    # Anomaly Count (Where error is 'large', e.g., > 3 std devs of recent window)
    # We re-calculate a simple Z-score here for the 'Tape'
    residuals = df['y_true'] - df['yhat']
    std = residuals.rolling(336).std().iloc[-1] # Last known volatility
    # Count anomalies in last 24h
    last_24h_resid = residuals.iloc[-24:]
    anomalies_today = (last_24h_resid.abs() > (3 * std)).sum()
    
    return mase, anomalies_today

# --- Dashboard Layout ---

# 1. Sidebar
st.sidebar.title("⚡ OPSD PowerDesk")
country = st.sidebar.selectbox("Select Country", ["DE", "FR", "ES"], index=0)
st.sidebar.markdown("---")
st.sidebar.markdown("**Simulation Status:**")

# Load Data
df, logs = load_data(country)

if df is None:
    st.error(f"No data found for {country}. Run the simulation (Day 11) first!")
    st.stop()

# 2. KPI Tiles (Top Row)
mase, anom_count = calculate_kpis(df)

# Get last update info
if not logs.empty:
    last_update_time = logs.iloc[0]['timestamp']
    last_reason = logs.iloc[0]['event']
else:
    last_update_time = "N/A"
    last_reason = "No Updates"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rolling 7D MASE", f"{mase:.2f}", delta_color="inverse") # Lower is better
col2.metric("Anomalies (Last 24h)", f"{anom_count}", delta_color="inverse")
col3.metric("Last Update Time", str(last_update_time).split('+')[0]) # Clean timestamp
col4.metric("Update Reason", last_reason)

# 3. Main Chart: Live Series & Forecast Cone
st.subheader(f"Real-Time Load Monitoring: {country}")

# Slider to zoom into recent history (Default: last 14 days = 336h)
window_hours = st.slider("History Window (Hours)", 24, 1000, 336)
subset = df.iloc[-window_hours:]

fig = go.Figure()

# Actual Load
fig.add_trace(go.Scatter(
    x=subset['timestamp'], y=subset['y_true'],
    name='Actual Load', line=dict(color='black', width=2)
))

# Forecast Line
fig.add_trace(go.Scatter(
    x=subset['timestamp'], y=subset['yhat'],
    name='Forecast', line=dict(color='#0000ff', width=2, dash='dot')
))

# "Anomaly Tape" - Highlight points with high error
# We calculate anomalies on the fly for visualization
residuals = subset['y_true'] - subset['yhat']
run_std = df['y_true'].sub(df['yhat']).rolling(336).std().iloc[-1] # Use global recent std
anoms = subset[residuals.abs() > 3*run_std]

fig.add_trace(go.Scatter(
    x=anoms['timestamp'], y=anoms['y_true'],
    mode='markers', name='Anomaly',
    marker=dict(color='red', size=8, symbol='x')
))

# "Forecast Cone" (Simulated for the dashboard demo)
# In a real app, we'd pull the 'lo'/'hi' cols. We'll estimate 80% PI here visually.
# Let's project the NEXT 24 hours from the end of data
last_time = subset['timestamp'].iloc[-1]
next_24h = pd.date_range(last_time, periods=25, freq='H')[1:]
# Use the last known value as a naive "mean" for the cone visual
dummy_mean = np.repeat(subset['yhat'].iloc[-1], 24)
# Widen the cone over time
cone_width = np.linspace(run_std, run_std*2, 24) 

fig.add_trace(go.Scatter(
    x=next_24h, y=dummy_mean + cone_width,
    mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
))
fig.add_trace(go.Scatter(
    x=next_24h, y=dummy_mean - cone_width,
    mode='lines', line=dict(width=0), fill='tonexty',
    fillcolor='rgba(0, 0, 255, 0.2)', name='Next 24h (80% PI)'
))

fig.update_layout(height=500, xaxis_title="Time (UTC)", yaxis_title="Load (MW)")
st.plotly_chart(fig, use_container_width=True)

# 4. Logs & Events
st.subheader("System Activity Log")
st.dataframe(logs, use_container_width=True)