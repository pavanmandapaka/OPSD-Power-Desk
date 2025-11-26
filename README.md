# OPSD PowerDesk - Electricity Load Forecasting

Time series forecasting for European electricity load using SARIMA and GRU models with online learning and anomaly detection.

## Countries

- **DE** - Germany
- **FR** - France
- **ES** - Spain

## Environment

### Setup

```bash
python3 -m venv venv

source venv/bin/activate  

pip install -r requirements.txt
```

### Requirements
- Python 
- pandas, numpy - Data manipulation
- statsmodels - SARIMA models
- tensorflow - GRU neural networks
- scikit-learn - ML anomaly detection
- streamlit, plotly - Dashboard
- tqdm - Progress bars

## How to Run

### 1. Run Simulations

```bash
source venv/bin/activate
python3 src/run_all_simulations.py

# Single country
python3 src/live_loop.py DE  # or FR, ES
```

### 2. Launch Dashboard

```bash
streamlit run src/dashboard_app.py
```

Access at `http://localhost:8501`

### 3. Calculate Metrics

```bash
python3 src/metrics.py
```

## Project Structure

```
├── config.yaml              # Configuration (countries, thresholds)
├── requirements.txt         # Python dependencies
├── data/                    # OPSD CSV files
├── src/
│   ├── load_opsd.py        # Data loading
│   ├── forecast.py         # SARIMA forecasting
│   ├── anomaly.py          # Z-score, CUSUM detection
│   ├── live_loop.py        # Online learning simulation
│   ├── dashboard_app.py    # Streamlit dashboard
│   └── metrics.py          # MASE, sMAPE, Coverage
├── outputs/                 # Generated forecasts and metrics
└── notebooks/               # Jupyter notebooks
```

## Performance

### SARIMA Models (Live Simulation)

| Country |  MASE | sMAPE | Coverage |
|---------|-------|-------|----------|
| DE      | 0.799 | 6.13% | 81.0%   |
| FR      | 0.976 | 5.03% | 84.6%   |
| ES      | 0.832 | 5.12% | 90.4%   |

### GRU Models 

| Country |  MASE | sMAPE | Coverage |
|---------|-------|-------|----------|
| DE      | 0.275 | 1.95% | 98.9%    |
| FR      | 0.288 | 1.41% | 65.3%    |
| ES      | 0.276 | 1.55% | 100.0%   |
