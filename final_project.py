import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────────────────
st.set_page_config(
    page_title="StockSight",
    page_icon="📈",
    layout="wide"
)

# ── Clean Title UI ──────────────────────────────────────
st.markdown("""
<h1 style='text-align: center; font-size: 48px;'>📈 StockSight</h1>
<p style='text-align: center; color: gray; margin-top: -10px;'>
Real-time Stock Analytics & ML Prediction
</p>
<hr>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    ticker = st.text_input("Ticker", "AAPL").upper()
    start_date = st.date_input("Start Date", datetime.today() - timedelta(days=365))
    end_date = st.date_input("End Date", datetime.today())

    forecast_days = st.slider("Forecast Days", 5, 60, 30)
    ma_short = st.slider("Short MA", 5, 50, 20)
    ma_long = st.slider("Long MA", 30, 200, 50)

    run_btn = st.button("Analyze")

# ── Data fetch ──────────────────────────────────────────
@st.cache_data(ttl=1800)
def get_data(ticker, start, end):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start, end=end)
    return df

# ── Safe fetch (fix rate limits) ────────────────────────
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

# ── Run app ─────────────────────────────────────────────
if run_btn or "loaded" not in st.session_state:
    st.session_state.loaded = True

    with st.spinner(f"Fetching {ticker} data..."):
        try:
            df = safe_fetch(ticker, start_date, end_date)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    if df.empty:
        st.error("Invalid ticker or no data.")
        st.stop()

    df = df.reset_index()

    # ── Moving averages ────────────────────────────────
    df["MA_Short"] = df["Close"].rolling(ma_short).mean()
    df["MA_Long"] = df["Close"].rolling(ma_long).mean()

    # ── Golden / Death Cross ──────────────────────────
    df["Signal"] = 0
    df.loc[ma_short:, "Signal"] = np.where(
        df["MA_Short"][ma_short:] > df["MA_Long"][ma_short:], 1, 0
    )
    df["Crossover"] = df["Signal"].diff()

    golden_cross = df[df["Crossover"] == 1]
    death_cross = df[df["Crossover"] == -1]

    # ── Chart ────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"],
        name="Price",
        line=dict(width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["MA_Short"],
        name="MA Short",
        line=dict(dash="dot")
    ))

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["MA_Long"],
        name="MA Long",
        line=dict(dash="dot")
    ))

    # Golden Cross markers
    fig.add_trace(go.Scatter(
        x=golden_cross["Date"],
        y=golden_cross["Close"],
        mode="markers",
        marker=dict(size=12, color="green", symbol="triangle-up"),
        name="Golden Cross (BUY)"
    ))

    # Death Cross markers
    fig.add_trace(go.Scatter(
        x=death_cross["Date"],
        y=death_cross["Close"],
        mode="markers",
        marker=dict(size=12, color="red", symbol="triangle-down"),
        name="Death Cross (SELL)"
    ))

    fig.update_layout(
        title=f"{ticker} Price with Trading Signals",
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── ML Model ─────────────────────────────────────
    df_ml = df.dropna().copy()
    df_ml["Days"] = (df_ml["Date"] - df_ml["Date"].min()).dt.days

    X = df_ml[["Days"]]
    y = df_ml["Close"]

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # ── Forecast ─────────────────────────────────────
    future_days_range = np.arange(
        X["Days"].max() + 1,
        X["Days"].max() + forecast_days + 1
    ).reshape(-1, 1)

    future_pred = model.predict(future_days_range)

    future_dates = [
        df_ml["Date"].max() + timedelta(days=i + 1)
        for i in range(forecast_days)
    ]

    # ── Prediction Chart ─────────────────────────────
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=df_ml["Date"], y=df_ml["Close"],
        name="Actual"
    ))

    fig2.add_trace(go.Scatter(
        x=df_ml["Date"].iloc[split:], y=y_pred,
        name="Predicted"
    ))

    fig2.add_trace(go.Scatter(
        x=future_dates, y=future_pred,
        name="Forecast"
    ))

    fig2.update_layout(
        template="plotly_dark",
        height=400
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.metric("Model RMSE", f"{rmse:.2f}")

    # ── Signal summary ───────────────────────────────
    if len(golden_cross) > 0 and golden_cross.index[-1] > death_cross.index[-1]:
        st.success("🟢 Latest Signal: BUY (Golden Cross)")
    elif len(death_cross) > 0:
        st.error("🔴 Latest Signal: SELL (Death Cross)")
    else:
        st.info("No recent crossover signal")
