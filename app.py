import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error

st.title("📈 Stock Price Forecaster")
st.caption("Exponential Smoothing vs. a Naive baseline. Educational only, not financial advice.")

ticker = st.text_input("Ticker symbol", "PFE").upper()
days = st.slider("Days to forecast", 5, 60, 30)


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

# Train/test split (80/20)
test_size = int(len(df) * 0.2)
train, test = df.iloc[:-test_size], df.iloc[-test_size:]

# Exponential Smoothing backtest
model = ExponentialSmoothing(train["close"], trend="add", seasonal="add", seasonal_periods=5).fit()
forecast = pd.Series(model.forecast(len(test)).values, index=test.index)
mae = mean_absolute_error(test["close"], forecast)

# Naive baseline: tomorrow = today
naive = test["close"].shift(1)
naive.iloc[0] = train["close"].iloc[-1]
naive_mae = mean_absolute_error(test["close"], naive)

col1, col2 = st.columns(2)
col1.metric("Exp. Smoothing MAE", f"${mae:.2f}")
col2.metric("Naive MAE", f"${naive_mae:.2f}")

if mae < naive_mae:
    st.success("Exponential Smoothing beats the naive baseline.")
else:
    st.warning("The naive baseline has a lower MAE.")

# Future forecast using all data
final = ExponentialSmoothing(df["close"], trend="add", seasonal="add", seasonal_periods=5).fit()
future_dates = pd.bdate_range(df.index[-1] + pd.offsets.BDay(1), periods=days)
future = pd.Series(final.forecast(days).values, index=future_dates)

st.subheader(f"Next {days} business days")
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(df["close"].iloc[-120:], label="Recent")
ax2.plot(future, "--", color="red", label="Forecast")
ax2.legend()
st.pyplot(fig2)
st.dataframe(future.round(2).rename("Forecast"))
