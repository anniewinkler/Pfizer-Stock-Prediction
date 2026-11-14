# Stock Price Forecaster (Streamlit)

Forecasts stock closing prices (default: Pfizer, PFE) using Holt-Winters Exponential Smoothing,
and compares it against a naive previous-day baseline using MAE.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Deployed on Streamlit Community Cloud from this repo (main file: `app.py`).

