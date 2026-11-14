import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Stock Price Forecaster", page_icon="📈", layout="wide")

st.title("Stock Price Forecaster")
st.caption(
    "Exponential Smoothing vs. a Naive (previous-day) baseline. "
)


#Data
@st.cache_data(ttl=3600, show_spinner="Downloading price data...")
def load_data(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if data is None or data.empty:
        return pd.DataFrame()
    # Newer yfinance versions return MultiIndex columns even for one ticker
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    df = data[["Close"]].reset_index()
    df.columns = ["date", "close"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").asfreq("B")      # business-day frequency
    df["close"] = df["close"].ffill()          # fill holidays
    return df.dropna()


def fit_hw(series: pd.Series) -> ExponentialSmoothing:
    # Same settings as the notebook: additive trend, additive weekly (5-day) seasonality
    return ExponentialSmoothing(
        series, trend="add", seasonal="add", seasonal_periods=5
    ).fit()


#Sidebar
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker symbol", value="PFE").strip().upper()
start = st.sidebar.date_input("Start date", value=dt.date(2019, 1, 1))
test_pct = st.sidebar.slider("Test set size (%)", 10, 40, 20, step=5)
horizon = st.sidebar.slider("Future forecast (business days)", 5, 120, 30)

if not ticker:
    st.info("Enter a ticker symbol in the sidebar.")
    st.stop()

df = load_data(ticker, start, dt.date.today() + dt.timedelta(days=1))
if df.empty or len(df) < 100:
    st.error(f"Not enough data for '{ticker}'. Check the ticker or start date.")
    st.stop()

#Backtest (mirrors the notebook)
test_size = int(len(df) * test_pct / 100)
train, test = df.iloc[:-test_size], df.iloc[-test_size:]

fitted = fit_hw(train["close"])
hw_forecast = pd.Series(fitted.forecast(len(test)).values, index=test.index)
hw_mae = mean_absolute_error(test["close"], hw_forecast)

naive = test["close"].shift(1)
naive.iloc[0] = train["close"].iloc[-1]
naive_mae = mean_absolute_error(test["close"], naive)

st.subheader(f"{ticker}: Backtest on the last {test_pct}% of data")
c1, c2, c3 = st.columns(3)
c1.metric("Exp. Smoothing MAE", f"${hw_mae:.3f}")
c2.metric("Naive baseline MAE", f"${naive_mae:.3f}")
c3.metric("Latest close", f"${df['close'].iloc[-1]:.2f}")

if hw_mae < naive_mae:
    st.success("Exponential Smoothing beats the naive baseline on this test set.")
else:
    st.warning(
        "The naive baseline (tomorrow = today) has a lower MAE. This is common for "
        "stocks: a multi-month forecast is compared with a one-day-ahead guess, "
        "and prices behave close to a random walk."
    )

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(train.index, train["close"], label="Train")
ax.plot(test.index, test["close"], label="Actual (test)")
ax.plot(hw_forecast.index, hw_forecast, "--", label=f"Exp. Smoothing (MAE {hw_mae:.2f})")
ax.set_xlabel("Date"); ax.set_ylabel("Price ($)"); ax.grid(True); ax.legend()
st.pyplot(fig)

n = min(30, len(test))
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(test.index[:n], test["close"].iloc[:n], marker="o", label="Actual")
ax2.plot(test.index[:n], hw_forecast.iloc[:n], "--", marker="x", label="Exp. Smoothing")
ax2.plot(test.index[:n], naive.iloc[:n], ":", marker="s", label="Naive (previous day)")
ax2.set_title(f"First {n} test days"); ax2.grid(True); ax2.legend()
st.pyplot(fig2)

#Future forecast
st.subheader(f"Forecast: next {horizon} business days")
full_fit = fit_hw(df["close"])
future_idx = pd.bdate_range(df.index[-1] + pd.offsets.BDay(1), periods=horizon)
future = pd.Series(full_fit.forecast(horizon).values, index=future_idx)

# Rough uncertainty band: in-sample residual std growing with sqrt(horizon)
resid_std = np.std(full_fit.resid)
band = resid_std * 1.96 * np.sqrt(np.arange(1, horizon + 1))

fig3, ax3 = plt.subplots(figsize=(12, 5))
recent = df["close"].iloc[-250:]
ax3.plot(recent.index, recent, label="Recent history")
ax3.plot(future.index, future, "--", color="tab:red", label="Forecast")
ax3.fill_between(future.index, future - band, future + band, color="tab:red", alpha=0.15,
                 label="Approx. 95% range")
ax3.set_xlabel("Date"); ax3.set_ylabel("Price ($)"); ax3.grid(True); ax3.legend()
st.pyplot(fig3)

out = pd.DataFrame({"Forecast": future.round(2),
                    "Low (approx.)": (future - band).round(2),
                    "High (approx.)": (future + band).round(2)})
out.index.name = "Date"
st.dataframe(out, width="stretch")
st.download_button("Download forecast as CSV", out.to_csv().encode(), f"{ticker}_forecast.csv", "text/csv")

st.caption(
    "Disclaimer: Stock prices are very hard to predict. This model is for coursework "
    "and learning only. Do not use it to make investment decisions."
)
