# Short-Term Ferry Ticket Demand Forecasting & Predictive Decision Support System

Unified Mentor internship project, sponsored by Toronto Government Parks, Forestry & Recreation.

Turns ten years of Toronto Island Park ferry ticket data (2015–2025) into short-term
(15 minute – 2 hour) demand forecasts, so operations staff get advance notice of
surges instead of finding out after the fact.

**Live dashboard:** [ferry-demand-forecasting-bpbtewsf9kmzjvhrfbtzvo.streamlit.app](https://ferry-demand-forecasting-bpbtewsf9kmzjvhrfbtzvo.streamlit.app/)
**Research paper:** [`docs/Ferry_Demand_Forecasting_Research_Paper.pdf`](docs/Ferry_Demand_Forecasting_Research_Paper.pdf)
**Executive summary:** [`docs/Ferry_Demand_Executive_Summary.pdf`](docs/Ferry_Demand_Executive_Summary.pdf)

## Key results

- Best model (Gradient Boosting) is **20–45% more accurate** than naive persistence,
  with the advantage *growing* at longer horizons — exactly where it's most useful
  operationally.
- 80% prediction intervals achieve **86–88% real coverage**, and adapt automatically:
  under 10 tickets wide on quiet days, 100+ on peak summer weekends.
- Following evaluator feedback (see below), adding holiday/seasonal features
  improved accuracy most at the *longest* forecast horizon: Gradient Boosting's
  2-hour-ahead MAE improved 3.9% (18.80 → 18.07 tickets) and Linear Regression's
  improved 10.2% (30.01 → 26.95) — both larger gains than at the 15-minute horizon,
  directly addressing the "longer horizons need better accuracy" note.
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

## Evaluator feedback (Unified Mentor, 3 Sept 2026 — 7.5/10, "Good")

The evaluator's two suggestions and how each was handled:

- **"Add weather, holidays, weekends, special events, seasonal patterns, and
  ferry schedule features."** Added what's directly computable from the
  existing timestamp data with no external dependency: statutory holiday and
  long-weekend flags (Canadian/Ontario calendar, computed algorithmically —
  see `CanadaOntarioHolidays` in `02_feature_engineering.py`), cyclical
  hour/day/month encodings, and same-time-yesterday / same-time-last-week
  seasonal lag features. **Not added:** real weather data and a special-events
  calendar — neither was available for this timestamp range in this
  development environment. Documented here rather than fabricated, consistent
  with how ARIMA/Prophet's unavailability is handled below.
- **"Improve accuracy at longer forecasting horizons."** The seasonal lag
  features specifically target this: recent (15–120 min) lags naturally lose
  predictive power the further ahead you forecast, but "what happened at this
  exact time last week" doesn't decay with horizon. Result: the 2-hour horizon
  saw the largest accuracy improvement of any horizon (see Key results above),
  confirming the fix landed where the feedback said it was needed.

## Notes & limitations

- ARIMA/SARIMA (statsmodels) and Prophet were in the original project scope but
  could not be evaluated in the development environment used for this phase
  (no internet access to install them). See the research paper, Section 10, for
  a recommended follow-up.
- Real weather data and a special-events calendar for Toronto Island Park are
  not included, for the same reason (no access to either in this environment) —
  noted as follow-up work rather than approximated.
- The dashboard currently visualizes backtested historical predictions, not a
  live ticketing feed.
