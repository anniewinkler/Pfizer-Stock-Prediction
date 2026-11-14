import datetime as dt

import pandas as pd
import streamlit as st
import yfinance as yf
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.title("📈 Pfizer Stock Price Predictor")
st.caption("Pick a future date and compare two methods.")

ticker = st.text_input("Ticker symbol", "PFE").upper()


@st.cache_data(ttl=3600)
def load_data(ticker):
    data = yf.download(ticker, start="2019-01-01", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    df = data[["Close"]].rename(columns={"Close": "close"})
    df = df.asfreq("B")
    df["close"] = df["close"].ffill()
    return df.dropna()


df = load_data(ticker)
if len(df) < 100:
    st.error("Not enough data. Check the ticker symbol.")
    st.stop()

last_date = df.index[-1].date()
last_close = float(df["close"].iloc[-1])
st.write(f"Latest close for **{ticker}**: **${last_close:.2f}** on {last_date}")

# Calendar dropdown
target = st.date_input(
    "Date to predict",
    value=last_date + dt.timedelta(days=30),
    min_value=last_date + dt.timedelta(days=1),
    max_value=last_date + dt.timedelta(days=365),
)

# Number of business days between the last data point and the target date
steps = len(pd.bdate_range(last_date + dt.timedelta(days=1), target))
if steps == 0:
    st.warning("That date falls on a weekend with no trading day in between. Pick a later date.")
    st.stop()
if target.weekday() >= 5:
    st.info("Markets are closed on weekends, so the prediction is for the prior Friday.")

# Method 1: Exponential Smoothing
model = ExponentialSmoothing(df["close"], trend="add", seasonal="add", seasonal_periods=5).fit()
es_pred = float(model.forecast(steps).iloc[-1])

# Method 2: Naive (assumes the price stays at the latest close)
naive_pred = last_close

st.subheader(f"Predicted price on {target}")
col1, col2 = st.columns(2)
col1.metric(
    "Exponential Smoothing",
    f"${es_pred:.2f}",
    f"{es_pred - last_close:+.2f} ({(es_pred / last_close - 1) * 100:+.1f}%)",
)
col2.metric(
    "Naive",
    f"${naive_pred:.2f}",
    f"{naive_pred - last_close:+.2f} ({(naive_pred / last_close - 1) * 100:+.1f}%)",
)
st.caption("Change shown is versus the latest close.")
