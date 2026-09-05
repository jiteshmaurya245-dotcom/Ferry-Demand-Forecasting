# Toronto Island Ferry — Demand Forecast Dashboard

## What this is
An interactive dashboard for the Short-Term Ferry Ticket Demand Forecasting project.
It shows backtested forecasts (Naive, Moving Average, Linear Regression, and Gradient
Boosting) across four horizons — 15 min, 30 min, 1 hour, 2 hours — with confidence
bands for the Gradient Boosting model, evaluated on a held-out 6-month test period
(2025-06-22 to 2025-12-21) the models never trained on.

## How to run it
1. Make sure you have Python 3.9+ installed.
2. In this folder, install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Launch the app:
   ```
   streamlit run app.py
   ```
4. It will open in your browser automatically (usually at http://localhost:8501).

## Files
- `app.py` — the dashboard
- `app_predictions.csv` — precomputed backtest predictions for all models/horizons
  (the app reads this rather than retraining models live, since training on the
  full 10-year dataset takes several minutes per model)
- `requirements.txt` — Python packages needed

## Notes / next steps
- This dashboard visualizes **backtested** predictions, not a live ticket feed —
  there's no real-time data source connected yet.
- Two libraries mentioned in the original brief — `statsmodels` (ARIMA/SARIMA) and
  `Prophet` — weren't available in the sandbox used to build this, so they're not
  included in the model comparison yet. Both are easy to add if you have internet
  access to `pip install` them: fit them the same way as the other models,
  save `Timestamp, target, prediction` to a CSV in the same shape as the others,
  and add a new entry to the `MODELS` dict in `app.py`.
