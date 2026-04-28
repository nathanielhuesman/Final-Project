import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────────────────
st.set_page_config(
    page_title="StockSight | Analytics & Prediction",
    page_icon="📈",
    layout="wide"
)

# ── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.title("📈 StockSight")

    ticker = st.text_input("Ticker", "AAPL").upper()

    start_date = st.date_input("Start Date", datetime.today() - timedelta(days=365))
    end_date = st.date_input("End Date", datetime.today())

    forecast_days = st.slider("Forecast Days", 5, 60, 30)
    ma_short = st.slider("Short MA", 5, 50, 20)
    ma_long = st.slider("Long MA", 30, 200, 50)

    run_btn = st.button("Analyze")

# ── Data fetch with cache ───────────────────────────────
@st.cache_data(ttl=1800)
def get_data(ticker, start, end):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start, end=end)
    info = stock.info
    return df, info

# ── Safe fetch (fix rate limit) ─────────────────────────
def safe_fetch(ticker, start, end):
    for i in range(3):
        try:
            return get_data(ticker, start, end)
        except Exception as e:
            if "Too Many Requests" in str(e):
                time.sleep(2 * (i + 1))
            else:
                raise e
    raise Exception("Rate limited. Try again.")

# ── Run only when button clicked ────────────────────────
if run_btn or "loaded" not in st.session_state:
    st.session_state.loaded = True

    with st.spinner(f"Fetching {ticker} data..."):
        try:
            df, info = safe_fetch(ticker, start_date, end_date)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    if df.empty:
        st.error("Invalid ticker or no data.")
        st.stop()

    df = df.reset_index()

    # ── Indicators ──────────────────────────────────────
    df["MA_Short"] = df["Close"].rolling(ma_short).mean()
    df["MA_Long"] = df["Close"].rolling(ma_long).mean()

    df["Return"] = df["Close"].pct_change()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # ── Chart ───────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Price"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA_Short"], name="MA Short"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA_Long"], name="MA Long"))

    st.plotly_chart(fig, use_container_width=True)

    # ── ML Model ────────────────────────────────────────
    df_ml = df.dropna()

    df_ml["Days"] = (df_ml["Date"] - df_ml["Date"].min()).dt.days

    X = df_ml[["Days"]]
    y = df_ml["Close"]

    split = int(len(X) * 0.8)

    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # ── Forecast ────────────────────────────────────────
    future_days_range = np.arange(X["Days"].max() + 1,
                                 X["Days"].max() + forecast_days + 1).reshape(-1, 1)

    future_pred = model.predict(future_days_range)

    future_dates = [df_ml["Date"].max() + timedelta(days=i+1) for i in range(forecast_days)]

    # ── Prediction Chart ────────────────────────────────
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(x=df_ml["Date"], y=df_ml["Close"], name="Actual"))
    fig2.add_trace(go.Scatter(x=df_ml["Date"].iloc[split:], y=y_pred, name="Predicted"))
    fig2.add_trace(go.Scatter(x=future_dates, y=future_pred, name="Forecast"))

    st.plotly_chart(fig2, use_container_width=True)

    st.metric("RMSE", f"{rmse:.2f}")

    # ── Signal ──────────────────────────────────────────
    if future_pred[-1] > df["Close"].iloc[-1]:
        st.success("BUY signal")
    else:
        st.error("SELL signal")
