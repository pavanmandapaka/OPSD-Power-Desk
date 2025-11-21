# OPSD PowerDesk Dashboard

## Overview
Interactive Streamlit dashboard for real-time monitoring of electricity load forecasting with SARIMA models.

## Features

### ✅ All Required Elements Implemented

#### i. Country Selector
- **Location:** Sidebar
- **Default:** Germany (DE) preselected as live country
- **Options:** DE, FR, ES
- **Purpose:** Switch between countries to view their live forecasts

#### ii. Live Series Chart
- **Display:** Last 7-14 days (configurable slider)
- **Lines:**
  - `y_true`: Actual load (blue solid line)
  - `yhat`: SARIMA forecast (orange dotted line)
- **Interactive:** Hover to see exact values, zoom, pan

#### iii. Forecast Cone
- **Projection:** Next 24 hours ahead from latest data point
- **Shading:** Orange transparent area showing 80% prediction interval (PI)
- **Mean Line:** Dashed orange line for expected forecast
- **Toggle:** Can be hidden/shown with checkbox

#### iv. Anomaly Tape
- **Markers:** Red X symbols on chart
- **Detection:** Uses `flag_z=1` from live forecasts (or calculates Z-score based anomalies)
- **Highlights:** Hours where forecast error exceeds 3 standard deviations
- **Toggle:** Can be hidden/shown with checkbox

#### v. KPI Tiles (4 metrics)
1. **Rolling 7D MASE**
   - Metric: Mean Absolute Scaled Error over last 168 hours
   - Interpretation: < 1.0 means better than naive forecast
   - Color: Green if good, red if needs attention

2. **80% PI Coverage (7D)**
   - Metric: Percentage of actual values within 80% prediction interval
   - Target: 75-85%
   - Tracks forecast calibration quality

3. **Anomalies (Last 24h)**
   - Count: Number of anomalous hours in past 24 hours
   - Threshold: Errors > 3σ (flag_z=1)
   - Alert: Highlights if count is high (> 5)

4. **Last Update Time**
   - Displays: Most recent model update timestamp
   - Format: YYYY-MM-DD HH:MM
   - Shows update reason (e.g., DRIFT_ADAPTATION, SCHEDULED_REFIT)

#### vi. Update Status Section
- **Location:** Bottom of dashboard
- **Latest Update:** Formatted timestamp and reason
- **Activity Log Table:** Shows recent 20 system events
  - Timestamp
  - Event type
  - Additional metadata
- **Source:** Reads from `{country}_online_updates.csv`

## Running the Dashboard

### Prerequisites
```bash
# Ensure you have run the simulations first
python3 src/run_all_simulations.py
```

### Start Dashboard
```bash
# Activate virtual environment
source venv/bin/activate

# Run Streamlit
streamlit run src/dashboard_app.py
```

### Access
- Local: http://localhost:8501
- Network: Available on your network IP (shown in terminal)

## Data Requirements

The dashboard expects the following files in `outputs/` directory:

1. **Forecast Files:**
   - `DE_live_forecasts.csv`
   - `FR_live_forecasts.csv`
   - `ES_live_forecasts.csv`
   
   Required columns:
   - `timestamp`: DateTime of forecast
   - `y_true`: Actual load value
   - `yhat`: Forecasted load value
   - `flag_z`: (Optional) Anomaly flag
   - `yhat_lower`, `yhat_upper`: (Optional) Confidence intervals

2. **Update Logs:**
   - `DE_online_updates.csv`
   - `FR_online_updates.csv`
   - `ES_online_updates.csv`
   
   Required columns:
   - `timestamp`: DateTime of update
   - `event`: Update reason/type

## Features in Detail

### Metrics Calculation

#### MASE (Mean Absolute Scaled Error)
```
MASE = MAE_forecast / MAE_naive
where MAE_naive uses 24-hour lag
```

#### PI Coverage
```
Coverage = (Count of y_true in [lower, upper]) / Total * 100%
```

#### Anomaly Detection
- Uses `flag_z` column if available
- Otherwise calculates rolling Z-scores with 336-hour window
- Threshold: |error| > 3 * rolling_std

### Interactive Controls

1. **History Window Slider**
   - Range: 7-30 days
   - Default: 14 days
   - Adjusts time range displayed on chart

2. **Show Anomalies Checkbox**
   - Toggles red anomaly markers
   - Default: ON

3. **Show Forecast Cone Checkbox**
   - Toggles 24h ahead projection
   - Default: ON

## Performance Notes

- Data is cached using `@st.cache_data`
- Dashboard auto-refreshes when files change
- Efficient for datasets with thousands of forecast points

## Troubleshooting

### No data found error
**Solution:** Run simulations first
```bash
python3 src/run_all_simulations.py DE
python3 src/run_all_simulations.py FR
python3 src/run_all_simulations.py ES
```

### Coverage shows N/A
**Cause:** Confidence interval columns missing
**Effect:** Dashboard estimates PI from historical errors

### Anomalies not showing
**Cause:** `flag_z` column missing
**Effect:** Dashboard calculates anomalies on-the-fly using Z-scores

## Future Enhancements

Potential additions:
- Multi-country comparison view
- Downloadable reports
- Alert configuration
- Historical performance trends
- Model comparison (SARIMA vs GRU)

## Files Modified

- `src/dashboard_app.py` - Main dashboard application
- Enhanced with all required features
- Uses modern Streamlit API (width='stretch' vs deprecated use_container_width)

## Dependencies

```
streamlit
pandas
numpy
plotly
```

All included in `requirements.txt`
