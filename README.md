# Short-Term Ferry Ticket Demand Forecasting & Predictive Decision Support System

Unified Mentor internship project, sponsored by Toronto Government Parks, Forestry & Recreation.

Turns ten years of Toronto Island Park ferry ticket data (2015–2025) into short-term
(15 minute – 2 hour) demand forecasts, so operations staff get advance notice of
surges instead of finding out after the fact.

**Live dashboard:** _add your Streamlit Cloud URL here after deploying_
**Research paper:** [`docs/Ferry_Demand_Forecasting_Research_Paper.pdf`](docs/Ferry_Demand_Forecasting_Research_Paper.pdf)
**Executive summary:** [`docs/Ferry_Demand_Executive_Summary.pdf`](docs/Ferry_Demand_Executive_Summary.pdf)

## Key results

- Best model (Gradient Boosting) is **20–45% more accurate** than naive persistence,
  with the advantage *growing* at longer horizons — exactly where it's most useful
  operationally.
- 80% prediction intervals achieve **86–88% real coverage**, and adapt automatically:
  under 10 tickets wide on quiet days, 100+ on peak summer weekends.
- Full methodology, figures, and discussion are in the research paper above.

## Repository structure

```
├── data/                    Raw dataset (cleaned/feature files are regenerated, not committed)
├── scripts/                 Reproducible pipeline, run in order
│   ├── 01_data_cleaning.py
│   ├── 02_feature_engineering.py
│   └── 03_train_evaluate_models.py
├── app/                     Streamlit dashboard (deployable as-is on Streamlit Cloud)
├── docs/                    Research paper & executive summary (PDF)
└── requirements.txt         Dependencies for the scripts/ pipeline
```

## Reproducing the analysis

```bash
pip install -r requirements.txt
python scripts/01_data_cleaning.py
python scripts/02_feature_engineering.py
python scripts/03_train_evaluate_models.py   # ~5-8 min; writes app/app_predictions.csv
```

## Running the dashboard locally

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

## Notes & limitations

- ARIMA/SARIMA (statsmodels) and Prophet were in the original project scope but
  could not be evaluated in the development environment used for this phase
  (no internet access to install them). See the research paper, Section 10, for
  a recommended follow-up.
- The dashboard currently visualizes backtested historical predictions, not a
  live ticketing feed.
